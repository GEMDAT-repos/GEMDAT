import pickle
from itertools import compress
from pathlib import Path
from typing import Optional, Sequence

from pymatgen.core import Lattice
from pymatgen.core.trajectory import Trajectory as PymatgenTrajectory
from pymatgen.io import vasp


class Trajectory(PymatgenTrajectory):

    def __init__(self, *, metadata: dict | None = None, **kwargs):
        super().__init__(**kwargs)
        self.metadata = metadata if metadata else None

    @classmethod
    def from_cache(cls, cache: str | Path):
        """Load data from cache using pickle.

        Parameters
        ----------
        cache : Path
            Name of cache file
        """
        with open(cache, 'rb') as f:
            obj = pickle.load(f)
        return obj

    def to_cache(self, cache: str | Path):
        """Dump data to cache using pickle.

        Parameters
        ----------
        cache : Path
            Name of cache file
        """
        with open(cache, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def from_vasprun(cls,
                     xml_file: str | Path,
                     cache: Optional[str | Path] = None):
        """Load data from vasprun.xml.

        Parameters
        ----------
        xml_file : Path
            Path to vasprun.xml
        cache : Optional[Path], optional
            Path to cache data for vasprun.xml

        Returns
        -------
        data : Data
            Dataclass with simulation data
        """
        if not cache:
            cache = Path(str(xml_file) + '.cache')

        if Path(cache).exists():
            try:
                return cls.from_cache(cache)
            except Exception as e:
                print(e)
                print('Error reading from cache, reading full VaspRun')

        run = vasp.Vasprun(
            xml_file,
            parse_dos=False,
            parse_eigen=False,
            parse_projected_eigen=False,
            parse_potcar_file=False,
        )

        metadata = {'temperature': run.parameters['TEBEG']}

        obj = cls.from_structures(
            run.structures,
            constant_lattice=True,
            time_step=run.parameters['POTIM'] * 1e-15,
            metadata=metadata,
        )
        obj.to_positions()

        if cache:
            obj.to_cache(cache)

        return obj

    def get_lattice(self, idx: int | None = None) -> Lattice:
        """Get lattice.

        Parameters
        ----------
        idx : int | None, optional
            Optionally, get lattice at specified index if the lattice is not constant

        Returns
        -------
        lattice : Lattice
            Pymatgen Lattice object
        """
        if self.constant_lattice:
            return Lattice(self.lattice)

        latt = self.lattices[idx]
        return Lattice(latt)

    def __getitem__(self, frames):
        """Hack around pymatgen Trajectory limitations."""
        new = super().__getitem__(frames)
        if isinstance(new, PymatgenTrajectory):
            new.__class__ = self.__class__
            new.metadata = self.metadata
        return new

    def filter(self, species: str | Sequence[str]):
        """Return trajectory with coordinates for specified species only.

        Parameters
        ----------
        speces : str | Sequence[str]
            Species to select, i.e. 'Li'

        Returns
        -------
        trajectory : Trajectory
            Output trajectory with coordinates for selected species only
        """
        idx = [sp.name in species for sp in self.species]
        new_coords = self.coords[:, idx]
        new_species = list(compress(self.species, idx))

        return self.__class__(species=new_species,
                              coords=new_coords,
                              lattice=self.get_lattice())
