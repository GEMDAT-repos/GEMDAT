from __future__ import annotations

import pytest
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from gemdat.io import get_list_of_known_materials, load_known_material, read_cif, write_cif
from gemdat.symmetry import SymmetryAnalyzer


def test_load_known_material():
    structure = load_known_material('argyrodite')
    assert isinstance(structure, Structure)
    assert all(label == '48h' for label in structure.labels)


def test_load_known_material_supercell():
    structure = load_known_material('argyrodite', supercell=(3, 2, 1))

    assert isinstance(structure, Structure)

    length = 9.924
    assert structure.lattice.a == 3 * length
    assert structure.lattice.b == 2 * length
    assert structure.lattice.c == 1 * length


def test_labels_supercell():
    structure = load_known_material('argyrodite', supercell=(1, 1, 2))
    assert isinstance(structure, Structure)
    assert all(label == '48h' for label in structure.labels)


def test_labels_multiple_species():
    structure = load_known_material('lisnps')
    assert isinstance(structure, Structure)
    assert set(structure.labels) == {'Li1', 'Li2', 'Li3', 'Li4'}


def test_get_list_of_known_materials():
    known_materials = get_list_of_known_materials()
    assert not any(name.endswith('.cif') for name in known_materials)


@pytest.fixture()
def noisy_structure():
    structure = Structure(
        lattice=Lattice.cubic(10.0),
        species=['P', 'S', 'Li', 'Li'],
        coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
    )
    structure.perturb(0.05, seed=0)
    return structure


def test_write_cif_idealised(noisy_structure, tmp_path):
    symprec = 0.3
    filename = tmp_path / 'idealised.cif'

    write_cif(noisy_structure, filename, symprec=symprec, idealised=True)

    reread = read_cif(filename)
    expected = SymmetryAnalyzer(noisy_structure).idealise(symprec)
    matcher = StructureMatcher(ltol=1e-4, stol=1e-4, angle_tol=1e-3, scale=False)
    assert matcher.fit(reread, expected)
    assert (
        SpacegroupAnalyzer(reread, symprec=1e-4).get_space_group_number()
        == SpacegroupAnalyzer(noisy_structure, symprec=symprec).get_space_group_number()
    )


def test_write_cif_idealised_needs_symprec(noisy_structure, tmp_path):
    with pytest.raises(ValueError, match='needs a `symprec`'):
        write_cif(noisy_structure, tmp_path / 'idealised.cif', idealised=True)


def _u_iso_column(filename):
    """`{label: U}` of the atom-site loop of a cif written by `write_cif`."""
    block = next(iter(CifParser(filename).as_dict().values()))
    return dict(zip(block['_atom_site_label'], map(float, block['_atom_site_U_iso_or_equiv'])))


def test_write_cif_u_iso(noisy_structure, tmp_path):
    symprec = 0.3
    noisy_structure.add_site_property('u_iso', [0.01, 0.02, 0.04, 0.06])
    plain = noisy_structure.copy()
    plain.remove_site_property('u_iso')

    write_cif(noisy_structure, tmp_path / 'u.cif', symprec=symprec)
    write_cif(plain, tmp_path / 'plain.cif', symprec=symprec)

    # one row per orbit, the two equivalent Li averaged
    assert _u_iso_column(tmp_path / 'u.cif') == pytest.approx(
        {'P1': 0.01, 'S1': 0.02, 'Li1': 0.05}
    )
    # the U column does not change the refined structure
    matcher = StructureMatcher(ltol=1e-6, stol=1e-6, angle_tol=1e-4, scale=False)
    assert matcher.fit(read_cif(tmp_path / 'u.cif'), read_cif(tmp_path / 'plain.cif'))


def test_write_cif_u_iso_without_symmetry(noisy_structure, tmp_path):
    noisy_structure.add_site_property('u_iso', [0.01, 0.02, 0.04, None])

    write_cif(noisy_structure, tmp_path / 'p1.cif')

    text = (tmp_path / 'p1.cif').read_text()
    assert text.count('?') == 1
    assert '0.04000' in text
