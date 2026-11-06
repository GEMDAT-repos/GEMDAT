"""This module contains functions to read and write data."""

from __future__ import annotations

import re
import warnings
from collections import Counter
from importlib.resources import files
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .symmetry import SymmetryAnalyzer

DATA = Path(files('gemdat') / 'data')  # type: ignore

SUPERCELL = {
    'argyrodite': (1, 1, 1),
    'lagp': (1, 2, 2),
    'latp': (1, 1, 1),
    'na3ps4': (2, 2, 2),
    'mno2_lambda': (1, 1, 1),
    'lisnps': (1, 1, 1),
    'li3ps4_beta': (1, 1, 2),
}


def write_cif(
    structure: Structure,
    filename: Path | str,
    *,
    idealised: bool = False,
    write_u_iso: bool = True,
    **kwargs,
):
    """Write structure to cif file using [pymatgen.io.cif.CifWriter][].

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
        Structure to save
    filename : Path | str
        Filename to write to
    idealised : bool
        If True, first move every site to its least-squares ideal position for
        the space group found at `symprec`, and idealise the lattice, see
        [SymmetryAnalyzer.idealise][gemdat.symmetry.SymmetryAnalyzer.idealise].
        Requires `symprec`. If False (default), the structure is passed on
        as is; note that [pymatgen.io.cif.CifWriter][] given a `symprec`
        still refines it with spglib, which places one representative site
        of every orbit and generates the rest of the orbit from it.
    write_u_iso : bool
        If True (default), write the `u_iso` site property of a structure
        that has one, see Notes. Set to False to leave it out.
    **kwargs : dict
        Additional keyword arguments passed to
        [pymatgen.io.cif.CifWriter][]. In particular, pass `symprec` to detect
        and write the space group symmetry (otherwise the cif is written in P1).

    Notes
    -----
    A `u_iso` site property (isotropic displacement parameter, Å², the same
    on every site of an orbit) is written as `_atom_site_U_iso_or_equiv`,
    with the sites labelled by orbit (`Li1`, `Li2`, ...). Orbits need a
    symmetry, so `symprec` then defaults to 0.01.

    Raises
    ------
    ValueError
        If `idealised` is set without a `symprec`, or no symmetry is found.
    """
    write_u_iso = write_u_iso and 'u_iso' in structure.site_properties
    if write_u_iso:
        kwargs.setdefault('symprec', 0.01)
    symprec = kwargs.get('symprec')
    angle_tolerance = kwargs.get('angle_tolerance', 5.0)

    if idealised:
        if symprec is None:
            raise ValueError('`idealised=True` needs a `symprec` to find the space group.')
        structure = SymmetryAnalyzer(structure, angle_tolerance=angle_tolerance).idealise(
            symprec
        )

    u_by_label: dict[str, float] = {}
    if write_u_iso:
        # `CifWriter` only writes site properties without symmetry, and its
        # refinement drops them, so label every orbit and look up the U of
        # each written row by its label.
        dataset = SpacegroupAnalyzer(
            structure, symprec=kwargs['symprec'], angle_tolerance=angle_tolerance
        ).get_symmetry_dataset()
        orbits = dataset.equivalent_atoms

        structure = structure.copy()
        count: Counter[str] = Counter()
        for orbit in dict.fromkeys(orbits.tolist()):
            symbol = structure[orbit].species.elements[0].symbol
            count[symbol] += 1
            label = f'{symbol}{count[symbol]}'
            for i in np.flatnonzero(orbits == orbit).tolist():
                structure[i].label = label
            u_by_label[label] = structure[orbit].properties['u_iso']

        # What `CifWriter` refines to (spglib's standardised cell), keeping
        # the labels: each standardised site is a translate of an input one.
        # Not `get_refined_structure(keep_site_properties=True)`: that gives
        # all sites of a species the property of its first site.
        source = [
            np.flatnonzero(dataset.mapping_to_primitive == p)[0]
            for p in dataset.std_mapping_to_primitive
        ]
        structure = Structure(
            dataset.std_lattice,
            [structure[i].species for i in source],
            dataset.std_positions,
            labels=[structure[i].label for i in source],
        )
        kwargs = {**kwargs, 'refine_struct': False}

    writer = CifWriter(structure, **kwargs)
    if write_u_iso:
        block = next(iter(writer.cif_file.data.values()))
        key = '_atom_site_U_iso_or_equiv'
        block.data[key] = [f'{u_by_label[lbl]:.5f}' for lbl in block.data['_atom_site_label']]
        next(loop for loop in block.loops if '_atom_site_label' in loop).append(key)
    writer.write_file(str(Path(filename).with_suffix('.cif')))


def _strip_label_suffixes(structure: Structure) -> None:
    """Strip pymatgen's `_1, _2, ...` site label suffixes in place.

    When expanding symmetry (on read) or building a supercell, pymatgen
    appends a numeric suffix to disambiguate the symmetry-equivalent copies
    of a labelled site (e.g. `48h` -> `48h_1, 48h_2, ...`). Gemdat groups the
    jump/transition analysis on these labels, so we strip the suffix to keep a
    shared label per crystallographic site.

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
        Structure to relabel in place
    """
    for site in structure:
        site.label = re.sub(r'_\d+$', '', site.label)


def read_cif(filename: Path | str) -> Structure:
    """Load cif file and return first item as
    [pymatgen.core.structure.Structure][].

    Parameters
    ----------
    filename : Path | str
        Filename of the structure in CIF format of

    Returns
    -------
    structure : pymatgen.core.structure.Structure
        Output structure
    """
    with warnings.catch_warnings():
        # Hide warning from https://github.com/materialsproject/pymatgen/pull/3419
        warnings.simplefilter('ignore')
        structure = Structure.from_file(filename, primitive=False)

    _strip_label_suffixes(structure)

    return structure


def load_known_material(
    name: str,
    supercell: tuple[int, int, int] | None = None,
) -> Structure:
    """Load known material from internal database.

    Parameters
    ----------
    name : str
        Name of the material
    supercell : tuple(int, int, int) | None, optional
        Optionally, scale the lattice by a sequence of three factors.
        For example, `(2, 1, 1)` specifies that the supercell should have
        dimensions $2a \\times b \\times c$.

    Returns
    -------
    structure : pymatgen.core.structure.Structure
        Output structure
    """
    filename = (DATA / name).with_suffix('.cif')

    if not filename.exists():
        raise ValueError(f'Unknown material: {name}')

    structure = read_cif(filename)

    if supercell:
        structure.make_supercell(supercell)
        _strip_label_suffixes(structure)

    return structure


def get_list_of_known_materials() -> list[str]:
    """Return list of known materials.

    Returns
    -------
    list[str]
        List with names of known materials
    """
    return [fn.stem for fn in DATA.glob('*.cif')]
