from pymatgen.core import Structure
from pymatgen.io import cif


def load_cif_as_structure(filename: str) -> Structure:
    """Load cif file and return first item as pymatgen structure."""
    cifdata = cif.CifParser(filename)
    structure = cifdata.get_structures(primitive=False)[0]
    return structure


def load_known_material(name: str) -> Structure:
    """Load known material."""
    filename = name
    return load_cif_as_structure(filename)
