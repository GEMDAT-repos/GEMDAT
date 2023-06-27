from gemdat.io import load_known_material
from pymatgen.core import Structure


def test_load_known_material():
    structure = load_known_material('argyrodite')

    assert isinstance(structure, Structure)
