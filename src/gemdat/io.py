"""This module contains functions to read and write data."""

from __future__ import annotations

import re
import warnings
from importlib.resources import files
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

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


def write_cif(structure: Structure, filename: Path | str, **kwargs):
    """Write structure to cif file using [pymatgen.io.cif.CifWriter][].

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
        Structure to save
    filename : Path | str
        Filename to write to
    **kwargs : dict
        Additional keyword arguments passed to
        [pymatgen.io.cif.CifWriter][]. In particular, pass `symprec` to detect
        and write the space group symmetry (otherwise the cif is written in P1).
    """
    filename = str(Path(filename).with_suffix('.cif'))
    CifWriter(structure, **kwargs).write_file(filename)


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
