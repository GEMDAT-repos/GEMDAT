from importlib.resources import files
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.io import cif

DATA = Path(files('gemdat') / 'data')  # type: ignore


def load_cif(filename: Path | str) -> Structure:
    """Load cif file and return first item as pymatgen structure."""
    cifdata = cif.CifParser(filename)
    structure = cifdata.get_structures(primitive=False)[0]
    return structure


def load_known_material(name: str) -> Structure:
    """Load known material."""
    filename = (DATA / name).with_suffix('.cif')

    if not filename.exists():
        raise ValueError(f'Unknown material: {name}')

    return load_cif(filename)
