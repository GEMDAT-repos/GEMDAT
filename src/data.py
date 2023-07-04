import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from pymatgen.core import Lattice, Species, Structure
from pymatgen.io import vasp

from .calculate.displacements import Displacements
from .calculate.vibration import Vibration


@dataclass(slots=True)
class SimulationData:
    """Dataclass to store simulation data."""

    structure: Structure
    trajectory_coords: np.ndarray
    species: Species
    lattice: Lattice
    time_step: float
    temperature: float
    parameters: dict[str, Any]
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_cache(cls, cache: str | Path):
        """Load data from cache using pickle.

        Parameters
        ----------
        cache : Path
            Name of cache file

        Returns
        -------
        data : SimulationData
            Dataclass with simulation data
        """
        with open(cache, 'rb') as f:
            data = pickle.load(f)
        return cls(**data)

    def to_cache(self, cache: str | Path):
        """Dump data to cache using pickle.

        Parameters
        ----------
        cache : Path
            Name of cache file
        """
        with open(cache, 'wb') as f:
            pickle.dump(asdict(self), f)

    def calculate_all(self, **kwargs):
        """Calculate extra parameters and place them in `.extras` attribute."""
        self.extras.update(
            Displacements.calculate_all(self, **kwargs, **self.extras))
        self.extras.update(
            Vibration.calculate_all(self, **kwargs, **self.extras))

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
        if cache and Path(cache).exists():
            return cls.from_cache(cache)

        run = vasp.Vasprun(
            xml_file,
            parse_dos=False,
            parse_eigen=False,
            parse_projected_eigen=False,
            parse_potcar_file=False,
        )

        structure = run.structures[0]
        trajectory = run.get_trajectory()
        trajectory.to_positions()

        data = {
            'structure': structure,
            'trajectory_coords': trajectory.coords,
            'species': structure.species,
            'lattice': structure.lattice,
            # size of the time step (*1e-15 = in femtoseconds)
            'time_step': run.parameters['POTIM'] * 1e-15,
            # temperature of the MD simulation
            'temperature': run.parameters['TEBEG'],
            'parameters': run.parameters,
        }

        # first create object to check pydantic before caching
        ret = cls(**data)

        if cache:
            ret.to_cache(cache)

        return ret
