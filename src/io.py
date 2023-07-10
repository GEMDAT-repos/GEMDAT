from importlib.resources import files
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.io import cif

DATA = Path(files('gemdat') / 'data')  # type: ignore


def load_cif(filename: Path | str) -> Structure:
    """Load cif file and return first item as pymatgen structure.

    Parameters
    ----------
    filename : Path | str
        Filename of the structure in CIF format of

    Returns
    -------
    structure : Structure
        Output structure
    """
    cifdata = cif.CifParser(filename)
    structure = cifdata.get_structures(primitive=False)[0]
    return structure


def load_known_material(name: str,
                        supercell: tuple[int, int, int] | None = None
                        ) -> Structure:
    """Load known material from internal database.

    Parameters
    ----------
    name : str
        Name of the material
    supercell : tuple(int, int, int) | None, optional
        Optionally, scale the lattice by a sequence of three factors.
        For example, (2, 1, 1) specifies that the supercell should have
        dimensions 2a x b x c.

    Returns
    -------
    structure : Structure
        Output structure
    """
    filename = (DATA / name).with_suffix('.cif')

    if not filename.exists():
        raise ValueError(f'Unknown material: {name}')

    structure = load_cif(filename)

    if supercell:
        structure.make_supercell(supercell)

    return structure
