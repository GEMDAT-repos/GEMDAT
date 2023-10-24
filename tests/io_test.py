from __future__ import annotations

from pymatgen.core import Structure

from gemdat.io import get_list_of_known_materials, load_known_material


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
