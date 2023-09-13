"""This module contains class for dealing with trajectories from molecular
dynamics simulations."""

from __future__ import annotations

import pickle
from itertools import compress
from pathlib import Path
from typing import Collection, Optional

import numpy as np
from pymatgen.core import Lattice
from pymatgen.core.trajectory import Trajectory as PymatgenTrajectory
from pymatgen.io import vasp


def _lengths(vectors: np.ndarray, lattice: Lattice) -> np.ndarray:
    """Calculate vector lengths using the metric tensor (Dunitz 1078, p227).

    Parameters
    ----------
    vectors : np.ndarray[i, j, k]
        Vectors in fractional coordinates
    metric_tensor : np.ndarray
        Metric tensor for the lattice

    Returns
    -------
    lengths : np.ndarray
        Vector lengths
    """
    metric_tensor = lattice.metric_tensor
    tmp = np.dot(vectors, metric_tensor)
    lengths_sq = np.einsum('ij,ji->i', tmp, vectors.T)
    assert lengths_sq.shape[0] == vectors.shape[0]
    assert lengths_sq.ndim == 1
    return np.sqrt(lengths_sq)


class Trajectory(PymatgenTrajectory):
    """Trajectory of sites from a molecular dynamics simulation."""

    def __init__(self, *, metadata: dict | None = None, **kwargs):
        """Initialize class.

        Parameters
        ----------
        metadata : dict | None, optional
            Optional dictionary with metadata
        **kwargs
            These are passed to [pymatgen.core.trajectory.Trajectory][]
        """
        super().__init__(**kwargs)
        self.metadata = metadata if metadata else {}

    def __getitem__(self, frames):
        """Hack around pymatgen Trajectory limitations."""
        new = super().__getitem__(frames)
        if isinstance(new, PymatgenTrajectory):
            new.__class__ = self.__class__
        new.metadata = self.metadata if hasattr(self, 'metadata') else {}
        return new

    def __repr__(self):
        base = self.get_structure(0)
        outs = [
            f'Full Formula ({base.composition.formula})',
            f'Reduced Formula: {base.composition.reduced_formula}',
        ]

        def to_str(x):
            return f'{x:>10.6f}'

        outs.append('abc   : ' + ' '.join(to_str(i) for i in base.lattice.abc))
        outs.append('angles: ' +
                    ' '.join(to_str(i) for i in base.lattice.angles))
        outs.append('pbc   : ' +
                    ' '.join(str(p).rjust(10) for p in base.lattice.pbc))
        if base._charge:
            outs.append(f'Overall Charge: {base._charge:+}')
        outs.append(f'Constant lattice ({self.constant_lattice})')
        outs.append(f'Sites ({len(base)})')
        outs.append(f'Time steps ({len(self)})')

        return '\n'.join(outs)

    def to_positions(self):
        """Pymatgen does not mod coords back to original unit cell.

        See [GEMDAT#103](https://github.com/GEMDAT-repos/GEMDAT/issues/103)
        """
        super().to_positions()
        self.coords = np.mod(self.coords, 1)

    @property
    def total_time(self) -> float:
        """Return total time for trajectory."""
        return len(self) * self.time_step

    @property
    def sampling_frequency(self) -> float:
        """Return number of time steps per second."""
        return 1 / self.time_step

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
    def from_cache(cls, cache: str | Path) -> Trajectory:
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
    def from_vasprun(
        cls,
        xml_file: str | Path,
        cache: Optional[str | Path] = None,
        **kwargs,
    ) -> Trajectory:
        """Load data from a `vasprun.xml`.

        Parameters
        ----------
        xml_file : Path
            Path to vasprun.xml
        cache : Optional[Path], optional
            Path to cache data for vasprun.xml
        **kwargs : dict
            Optional arguments passed to [pymatgen.io.vasp.outputs.Vasprun][]

        Returns
        -------
        trajectory : Trajectory
            Output trajectory
        """
        kwargs.setdefault('parse_dos', False)
        kwargs.setdefault('parse_eigen', False)
        kwargs.setdefault('parse_projected_eigen', False)
        kwargs.setdefault('parse_potcar_file', False)

        if not cache:
            cache = Path(str(xml_file) + '.cache')

        if Path(cache).exists():
            try:
                return cls.from_cache(cache)
            except Exception as e:
                print(e)
                print('Error reading from cache, reading full VaspRun')

        run = vasp.Vasprun(xml_file, **kwargs)

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
        lattice : pymatgen.core.lattice.Lattice
            Pymatgen Lattice object
        """
        if self.constant_lattice:
            return Lattice(self.lattice)

        latt = self.lattices[idx]
        return Lattice(latt)

    @property
    def cumulative_displacements(self) -> np.ndarray:
        """Return cumulative displacement vectors from base positions.

        This differs from [displacements()][gemdat.trajectory.Trajectory.displacements]
        in that it ignores the periodic boundary
        conditions. Instead, it cumulatively tracks the lattice translation vector (jimage).
        """
        return np.cumsum(self.displacements, axis=0)

    def distances_from_base_position(self) -> np.ndarray:
        """Return total distances from base positions.

        Ignores periodic boundary conditions.
        """
        lattice = self.get_lattice()

        all_distances = []

        for diff_vectors in self.cumulative_displacements:
            distances = _lengths(diff_vectors, lattice=lattice)
            all_distances.append(distances)

        all_distances = np.array(all_distances).T

        return all_distances

    def drift(
        self,
        *,
        fixed_species: None | str | Collection[str] = None,
        floating_species: None | str | Collection[str] = None,
    ) -> np.ndarray:
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
    ) -> Trajectory:
        """Apply drift correction to trajectory. For details see
        [drift()][gemdat.trajectory.Trajectory.drift].

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
                              base_positions=self.base_positions,
                              time_step=self.time_step)

    def filter(self, species: str | Collection[str]) -> Trajectory:
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
                              metadata=self.metadata,
                              time_step=self.time_step)
