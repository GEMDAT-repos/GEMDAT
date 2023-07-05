import pickle
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
from pymatgen.core import Lattice, Species, Structure
from pymatgen.io import vasp

from .calculate.displacements import Displacements
from .calculate.tracer import Tracer
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
        # convert to dict without object conversion
        with open(cache, 'wb') as f:
            pickle.dump(self.dict, f)

    @property
    def dict(self) -> dict:
        """Dict is an alternative for dataclasses.asdict.

        dataclasses.asdict has a problem with pymatgenclasses, so we do
        a more superficial conversion
        """
        return {
            slotname: getattr(self, slotname)
            for slotname in self.__slots__  # type: ignore
        }

    def calculate_all(self,
                      *,
                      diffusing_element: str,
                      known_structure: str | None = None,
                      equilibration_steps: int = 1250,
                      diffusion_dimensions: int = 3,
                      z_ion: float = 1.0,
                      n_parts: int = 10,
                      dist_collective: float = 4.5):
        """Calculate extra parameters and return them as a simple namespace.

        Parameters
        ----------
        diffusing_element : str
            Name of the diffusing element
        structure : str | None
            Path to cif file or name of known structure
        equilibration_steps : int
            Number of equilibration steps
        diffusion_dimensions : int
            Number of diffusion dimensions
        z_ion : float
            Ionic charge of the diffusing ion
        n_parts : int
            In how many parts to divide your simulation for statistics
        dist_collective : float
            Maximum distance for collective motions in Angstrom
        """
        extras = SimpleNamespace(
            diffusing_element=diffusing_element,
            known_structure=known_structure,
            equilibration_steps=equilibration_steps,
            diffusion_dimensions=diffusion_dimensions,
            z_ion=z_ion,
            n_parts=n_parts,
            dist_collective=dist_collective,
        )

        self._add_shared_variables(extras)

        extras.__dict__.update(Displacements.calculate_all(self,
                                                           extras=extras))
        extras.__dict__.update(Vibration.calculate_all(self, extras=extras))
        extras.__dict__.update(Tracer.calculate_all(self, extras=extras))

        return extras

    def _add_shared_variables(self, extras: SimpleNamespace):
        """Add common shared variables to extras namespace."""
        extras.n_diffusing = sum(
            [e.name == extras.diffusing_element for e in self.species])
        extras.n_steps = len(
            self.trajectory_coords) - extras.equilibration_steps

        diffusing_idx = np.argwhere([
            e.name == extras.diffusing_element for e in self.species
        ]).flatten()

        extras.diff_coords = self.trajectory_coords[
            extras.equilibration_steps:, diffusing_idx, :]

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
