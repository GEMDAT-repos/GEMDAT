from gemdat.io import load_known_material
from pymatgen.core import Structure


def test_load_known_material():
    structure = load_known_material('argyrodite')

    assert isinstance(structure, Structure)


def test_load_known_material_supercell():
    structure = load_known_material('argyrodite', supercell=(3, 2, 1))

    assert isinstance(structure, Structure)

    length = 9.924
    assert structure.lattice.a == 3 * length
    assert structure.lattice.b == 2 * length
    assert structure.lattice.c == 1 * length
