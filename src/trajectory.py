import pickle
from itertools import compress
from pathlib import Path
from typing import Collection, Optional

import numpy as np
from pymatgen.core import Lattice
from pymatgen.core.trajectory import Trajectory as PymatgenTrajectory
from pymatgen.io import vasp


class Trajectory(PymatgenTrajectory):

    def __init__(self, *, metadata: dict | None = None, **kwargs):
        super().__init__(**kwargs)
        self.metadata = metadata if metadata else None

    @property
    def positions(self) -> np.ndarray:
        """Return trajectory positions as fractional coordinates.

        Returns
        -------
        positions : np.ndarray
            Output array with positions
        """
        self.to_positions()
        return self.coords

    @property
    def displacements(self) -> np.ndarray:
        """Return trajectory displacements as fractional coordinates from base
        position.

        Returns
        -------
        displacements : np.ndarray
            Output array with displacements
        """
        self.to_displacements()
        return self.coords

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

    @staticmethod
    def lengths(vectors: np.ndarray, metric_tensor: np.ndarray) -> np.ndarray:
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
            Vector lengths
        """
        tmp = np.dot(vectors, metric_tensor)
        total_displacement = np.einsum('ij,ji->i', tmp, vectors.T)
        assert total_displacement.shape[0] == vectors.shape[0]
        assert total_displacement.ndim == 1
        return np.sqrt(total_displacement)

    @property
    def total_displacements(self) -> np.ndarray:
        """Return total displacement vectors from base position.

        This differs from `.displacements` in that it ignores the periodic boundary
        conditions. Instead, it cumulatively tracks the lattice translation vector (jimage).
        """
        return np.cumsum(self.displacements, axis=0)

    def total_distances(self) -> np.ndarray:
        """Return total distance from base position."""
        lattice = self.get_lattice()

        all_distances = []

        for diff_vectors in self.total_displacements:
            distances = Trajectory.lengths(diff_vectors,
                                           metric_tensor=lattice.metric_tensor)
            all_distances.append(distances)

        all_distances = np.array(all_distances).T

        return all_distances

    def drift(
        self,
        *,
        fixed_species: None | str | Collection[str] = None,
        floating_species: None | str | Collection[str] = None,
    ):
        """Compute drift by averaging the displacement from the base positions
        per frame.

        If no species are specified, use all species to calculate drift.

        Only one of `fixed_species` and `floating_species` should be specified.

        Parameters
        ----------
        fixed_species : None | str | Collection[str]
            These species are assumed fixed, and are used to calculate drift (e.g. framework species).
        floating_species : None | str | Collection[str]
            These species are assumed floating, and is used to determine the fixed species.

        Returns
        -------
        drift : np.array
            Output array with average drift per frame.
        """
        if fixed_species:
            displacements = self.filter(species=fixed_species).displacements
        elif floating_species:
            species = {
                sp.symbol
                for sp in self.species if sp.symbol not in floating_species
            }
            displacements = self.filter(species=species).displacements
        else:
            displacements = self.displacements

        return np.mean(displacements, axis=1)[:, None, :]

    def apply_drift_correction(
        self,
        *,
        fixed_species: None | str | Collection[str] = None,
        floating_species: None | str | Collection[str] = None,
    ):
        """Apply drift correction to trajectory. For details see `Trajectory.drift`.

        If no species are specified, use all species to calculate drift.

        Only one of `fixed_species` and `floating_species` should be specified.

        Parameters
        ----------
        fixed_species : None | str | Collection[str]
            These species are assumed fixed, and are used to calculate drift (e.g. framework species).
        floating_species : None | str | Collection[str]
            These species are assumed floating, and is used to determine the fixed species.

        Returns
        -------
        trajectory : Trajectory
            Ouput trajectory with positions corrected for drift
        """
        drift = self.drift(fixed_species=fixed_species,
                           floating_species=floating_species)

        return self.__class__(species=self.species,
                              coords=self.displacements - drift,
                              lattice=self.get_lattice(),
                              metadata=self.metadata,
                              coords_are_displacement=True,
                              base_positions=self.base_positions)

    def filter(self, species: str | Collection[str]):
        """Return trajectory with coordinates for given species only.

        Parameters
        ----------
        species : str | Collection[str]
            Species to select, i.e. 'Li'

        Returns
        -------
        trajectory : Trajectory
            Output trajectory with coordinates for selected species only
        """
        idx = [sp.symbol in species for sp in self.species]
        new_coords = self.positions[:, idx]
        new_species = list(compress(self.species, idx))

        return self.__class__(species=new_species,
                              coords=new_coords,
                              lattice=self.get_lattice(),
                              metadata=self.metadata)
