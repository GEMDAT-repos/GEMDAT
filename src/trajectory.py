import copy
import pickle
from functools import lru_cache
from itertools import compress
from pathlib import Path
from typing import Optional

import numpy as np
from pymatgen.core import Lattice
from pymatgen.core.trajectory import Trajectory as PymatgenTrajectory
from pymatgen.core.units import FloatWithUnit
from pymatgen.io import vasp
from scipy.constants import Avogadro, Boltzmann, angstrom, elementary_charge


class Trajectory(PymatgenTrajectory):

    def __init__(self, diffusing_element=None, **kwargs):
        super().__init__(**kwargs)
        self.diffusing_element = diffusing_element

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
                     cache: Optional[str | Path] = None,
                     **kwargs):
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

        obj = cls.from_structures(
            run.structures,
            constant_lattice=True,
            time_step=run.parameters['POTIM'] * 1e-15,
            **kwargs,
        )
        obj.to_positions()
        obj.temperature = run.parameters['TEBEG']

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
            return Lattice(self.lattice)  # type: ignore

        latt = self.lattices[idx]
        return Lattice(latt)

    def _cell_offsets(self) -> np.ndarray:
        """Calculate how many time the atoms wrap around the supercell with
        relation to the first position. This is needed to calculate the total
        displacment over time.

        For example, if a site is at [0, 0, 0.9] -> [0, 0, 0.1]
        assume it has jumped to the next cell: [0, 0, 1.1]

        Parameters
        ----------
        trajectory : Trajectory
            Input trajectory

        Returns
        -------
        offsets : np.ndarray[i, j, k]
            Integer array with unit cell offset vectors.
        """
        coords = self.coords

        first = coords[0, np.newaxis]
        diff = np.diff(coords, axis=0, prepend=first)

        digits = np.digitize(diff, bins=[0.5, -0.4999999]) - 1

        offsets = np.cumsum(digits, axis=0)
        return offsets

    def _lengths(self, vectors: np.ndarray) -> np.ndarray:
        """Calculate vector lengths using the metric tensor (Dunitz 1078,
        p227).

        Parameters
        ----------
        vectors : np.ndarray[i, j, k]
            Vectors as in fractional coordinates
        metric_tensor : np.ndarray
            Metric tensor for the lattice

        Returns
        -------
        lengths : np.ndarray
            Vector lengths in cartesian coordinates
        """
        tmp = np.dot(vectors, self.get_lattice().metric_tensor)
        total_displacement = np.einsum('ij,ji->i', tmp, vectors.T)
        assert total_displacement.shape[0] == vectors.shape[0]
        assert total_displacement.ndim == 1
        return np.sqrt(total_displacement)

    @lru_cache
    def displacements(self) -> np.ndarray:
        """Calculate displacements from first set of positions.

        Corrects for elements jumping to the next unit cell.

        TODO: figure out if this is better or PymatgenTrajectory.to_displacements()

        Returns
        -------
        displacements : np.ndarray[i, j]
            Displacements from first set of positions, in cartesian coordinates.
        """
        self.get_lattice()

        offsets = self._cell_offsets()

        corrected_coords = self.coords + offsets

        displacements = []

        first = corrected_coords[0]

        for disp in corrected_coords:
            diff_vectors = disp - first
            lengths = self._lengths(diff_vectors)
            displacements.append(lengths)

        displacements = np.array(displacements)

        displacements = displacements.T

        return displacements

    @property
    def total_time(self):
        return len(self.trajectory) * self.time_step

    @lru_cache
    def particle_density(self) -> FloatWithUnit:
        """Function that returns the particle density. when a diffusing element
        is specified, it returns the particle density of that element.

        Returns
        -------
        FloatWithUnit
        """

        n_diffusing = len(self.species)

        volume_ang = self.lattice.volume  # type: ignore

        volume_m3 = volume_ang * angstrom**3
        return FloatWithUnit(n_diffusing / volume_m3, 'm^-3')

    @property
    def mol_per_liter(self) -> FloatWithUnit:
        """mol_per_liter. when a diffusing element is specified, this function
        returns the mol per liter for specified element.

        Returns
        -------
        FloatWithUnit
        """

        mol_per_liter = (self.particle_density() * 1e-3) / Avogadro
        return FloatWithUnit(mol_per_liter, 'mol l^-1')

    def tracer_diffusivity(self, diffusion_dimensions=3) -> FloatWithUnit:
        """Calculate the tracer diffusivity, optionally for a specified
        element.

        Parameters
        ----------
        diffusion_dimensions :
            diffusion_dimensions

        Returns
        -------
        FloatWithUnit m^2 s^-1
        """

        msd = np.mean(self.displacements()[:, -1]**2)  # Angstrom^2

        # Diffusivity = MSD/(2*dimensions*time)
        tracer_diffusivity = (msd * angstrom**2) / (2 * diffusion_dimensions *
                                                    self.total_time)
        return FloatWithUnit(tracer_diffusivity, 'm^2 s^-1')

    def tracer_conductivity(self,
                            diffusion_dimensions=3,
                            z_ion: float = 1.0) -> FloatWithUnit:
        """tracer_conductivity,

        Parameters
        ----------
        diffusion_dimensions :
            diffusion_dimensions
        z_ion : float
            Ionic charge of the diffusing ion

        Returns
        -------
        FloatWithUnit
        """
        # Conductivity = elementary_charge^2 * charge_ion^2 * diffusivity * particle_density / (k_B * T)
        tracer_conduc = (
            (elementary_charge**2) *
            (z_ion**2) * self.tracer_diffusivity(diffusion_dimensions) *
            self.particle_density()) / (Boltzmann * self.temperature)
        return FloatWithUnit(tracer_conduc, 'S m^-1')

    def __getitem__(self, frames):
        """Hack around pymatgen Trajectory limitations."""
        new = super().__getitem__(frames)
        if isinstance(new, PymatgenTrajectory):
            new.__class__ = self.__class__
            new.temperature = self.temperature
        return new

    def filter(self, diffusing_element: str):
        """Returns a trajectory with only the diffusing element, useful for
        some plots.

        Could be extended to be a more generic filter.

        Parameters
        ----------
        diffusing_element : str
            diffusing_element

        Returns
        -------
        Trajectory
        """
        new = copy.deepcopy(self)

        idx = np.argwhere([
            e == diffusing_element
            if isinstance(e, str) else e.name == diffusing_element
            for e in self.species
        ])
        new.coords = self.coords[:, idx].squeeze()
        new.species = list(compress(self.species, idx))
        return new
