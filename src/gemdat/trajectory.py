"""This module contains class for dealing with trajectories from molecular
dynamics simulations."""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import xml.etree.ElementTree as ET
from itertools import compress, pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Collection, Optional

import numpy as np
import pandas as pd
from pymatgen.core import Element, Lattice, Species
from pymatgen.core.trajectory import Trajectory as PymatgenTrajectory
from pymatgen.io import vasp

from ._plot_backend import plot_backend

if TYPE_CHECKING:
    from pymatgen.core import Structure

    from .metrics import TrajectoryMetrics
    from .rdf import RDFData
    from .transitions import Transitions
    from .volume import Volume


SP_NAME = re.compile(r'([a-zA-Z]+)')


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
        outs.append('angles: ' + ' '.join(to_str(i) for i in base.lattice.angles))
        outs.append('pbc   : ' + ' '.join(str(p).rjust(10) for p in base.lattice.pbc))
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

    def to_volume(self, resolution: float = 0.2) -> Volume:
        """Calculate density volume from a trajectory.

        All coordinates are binned into voxels. The value of each
        voxel represents the number of coodinates that are associated
        with it.

        For more info, see [gemdat.Volume][].

        Parameters
        ----------
        resolution : float, optional
            Minimum resolution for the voxels in Angstrom

        Returns
        -------
        vol : Volume
            Output volume
        """
        from gemdat.volume import trajectory_to_volume

        return trajectory_to_volume(self, resolution=resolution)

    @property
    def time_step_ps(self) -> float:
        """Return time step in picoseconds."""
        assert self.time_step
        return self.time_step * 1e12

    @property
    def total_time(self) -> float:
        """Return total time for trajectory."""
        assert self.time_step
        return len(self) * self.time_step

    @property
    def sampling_frequency(self) -> float:
        """Return number of time steps per second."""
        assert self.time_step
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
        constant_lattice: bool = True,
        **kwargs,
    ) -> Trajectory:
        """Load data from a `vasprun.xml`.

        Parameters
        ----------
        xml_file : Path
            Path to vasprun.xml
        cache : Optional[Path], optional
            Path to cache data for vasprun.xml
        constant_lattice : bool
            Whether the lattice changes during the simulation,
            such as in an NPT MD simulation.
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
            serialized = json.dumps(kwargs, sort_keys=True).encode()
            hashid = hashlib.sha1(serialized).hexdigest()[:8]
            cache = Path(xml_file).with_suffix(f'.xml.{hashid}.cache')

        if Path(cache).exists():
            try:
                return cls.from_cache(cache)
            except Exception as e:
                print(e)
                print(f'Error reading from cache, reading {xml_file!r}')
        try:
            run = vasp.Vasprun(xml_file, **kwargs)
        except ET.ParseError as e:
            raise Exception(
                f'Error parsing {xml_file!r}, to parse incomplete data '
                'adding `exception_on_bad_xml=False` might help'
            ) from e

        metadata = {'temperature': run.parameters['TEBEG']}

        obj = cls.from_structures(
            run.structures,
            constant_lattice=constant_lattice,
            time_step=run.parameters['POTIM'] * 1e-15,
            metadata=metadata,
        )
        obj.to_positions()

        if cache:
            obj.to_cache(cache)

        return obj

    @classmethod
    def from_lammps(
        cls,
        *,
        coords_file: Path | str,
        data_file: Path | str,
        temperature: float,
        time_step: float,
        coords_format: str = 'xyz',
        atom_style: str = 'atomic',
        type_mapping: Optional[dict[str, str]] = None,
        cache: Optional[str | Path] = None,
        constant_lattice: bool = True,
    ) -> Trajectory:
        """Load data from LAMMPS.

        Parameters
        ----------
        coords_file : Path | str
            LAMMPS coords file with trajectory data
        data_file : Path | str
            LAMMPS data file with the lattice
        temperature : float
            Temperature of simulation in K
        time_step : float
            Time step of the simulation in ps
        coords_format: str
            Format of the coords file
        atom_style : str
            Atom style for box file
        type_mapping: dict[str, str], optional
            If specified, map numbers to element names. This is for LAMMPS
            data that do not contain element labels for the different species.
            See: https://github.com/GEMDAT-repos/GEMDAT/issues/353
        cache : Optional[Path], optional
            Path to cache data for vasprun.xml
        constant_lattice : bool
            Whether the lattice changes during the simulation,
            such as in an NPT MD simulation.

        Returns
        -------
        trajectory : Trajectory
            Output trajectory
        """
        from MDAnalysis import Universe
        from pymatgen.io.lammps.data import LammpsData

        coords_file = str(coords_file)
        data_file = str(data_file)

        if not cache:
            kwargs = {
                'coords_file': coords_file,
                'data_file': data_file,
                'temperature': temperature,
                'time_step': time_step,
            }
            serialized = json.dumps(kwargs, sort_keys=True).encode()
            hashid = hashlib.sha1(serialized).hexdigest()[:8]
            cache = Path(coords_file).with_suffix(f'.{coords_format}.{hashid}.cache')

        if Path(cache).exists():
            try:
                return cls.from_cache(cache)
            except Exception as e:
                print(e)
                print(f'Error reading from cache, reading {coords_file!r}')

        if not constant_lattice:
            raise NotImplementedError('Lammps reader does not support NPT simulations')

        try:
            lammps_data = LammpsData.from_file(filename=data_file, atom_style=atom_style)
        except pd.errors.ParserError as exc:
            msg = (
                f"Could not parse LAMMPS data file '{data_file}'."
                '\nSuggestion: Export the data file directly from LAMMPS'
                ' using the `write_data` command.'
            )
            raise IOError(msg) from exc

        lammps_data = LammpsData.from_file(filename=data_file, atom_style=atom_style)
        lattice = lammps_data.structure.lattice

        utraj = Universe(coords_file, format=coords_format)
        coords = utraj.trajectory.timeseries()
        coords = lattice.get_fractional_coords(coords)

        if type_mapping:
            species = [Element(type_mapping.get(_type)) for _type in utraj.atoms.types]  # type: ignore
        else:
            species = [Element(sp) for sp in utraj.atoms.elements]

        obj = cls(
            species=species,
            coords=coords,
            lattice=lattice,
            time_step=time_step * 1e-12,  # ps -> s
            constant_lattice=constant_lattice,
            metadata={'temperature': temperature},
        )
        obj.to_positions()

        if cache:
            obj.to_cache(cache)

        return obj

    @classmethod
    def from_gromacs(
        cls,
        *,
        topology_file: Path | str,
        coords_file: Path | str,
        constant_lattice: bool = True,
        temperature: float,
        extract_edr: bool = False,
        edr_file: Optional[str | Path] = None,
        cache: Optional[str | Path] = None,
    ) -> Trajectory:
        """Load data from GROMACS.

        Parameters
        ----------
        topology_file : Path | str
            GROMACS topology data file
        coords_file : Path | str
            GROMACS trajectory data file
        constant_lattice : bool
            Whether the lattice changes during the simulation,
            such as in an NPT MD simulation.
        temperature : float
            Temperature of simulation in K
        edr_file: Optional[Path], optional
            If specified, extract time-series energy data from GROMACS EDR file
        cache : Optional[Path], optional
            Path to cache data for vasprun.xml

        Returns
        -------
        trajectory : Trajectory
            Output trajectory
        """

        import MDAnalysis as mda
        from pymatgen.core import Lattice

        topology_file = str(topology_file)
        coords_file = str(coords_file)

        if not cache:
            kwargs = {
                'topology_file': topology_file,
                'coords_file': coords_file,
                'edr_file': edr_file,
                'temperature': temperature,
            }
            serialized = json.dumps(kwargs, sort_keys=True).encode()
            hashid = hashlib.sha1(serialized).hexdigest()[:8]
            cache = Path(coords_file).with_suffix(f'.{hashid}.cache')

        if Path(cache).exists():
            try:
                return cls.from_cache(cache)
            except Exception as e:
                print(e)
                print(f'Error reading from cache, reading {coords_file!r}')

        utraj = mda.Universe(topology_file, coords_file)
        coords = utraj.trajectory.timeseries()

        if not constant_lattice:
            lattices = [Lattice.from_parameters(*ts.dimensions) for ts in utraj.trajectory]
            for ts, lat in enumerate(lattices):
                coords[ts, :, :] = lat.get_fractional_coords(coords[ts, :, :])
        else:
            lattice = Lattice.from_parameters(*utraj.trajectory[0].dimensions)
            coords = lattice.get_fractional_coords(coords)

        species = [Element(SP_NAME.match(sp).group().capitalize()) for sp in utraj.atoms.names]  # type: ignore

        site_properties = {
            'residue': [sp.residue for sp in utraj.atoms],
            'residue_name': [sp.resname for sp in utraj.atoms],
            'residue_id': [sp.resid for sp in utraj.atoms],
        }

        metadata = {
            'temperature': temperature,
        }

        if edr_file:
            edr_file = str(edr_file)
            aux = mda.auxiliary.EDR.EDRReader(edr_file)
            metadata['aux'] = aux.get_data(aux.terms)

        obj = cls(
            species=species,
            coords=coords,
            lattice=lattice,
            time_step=utraj.trajectory.dt * 1e-12,  # ps -> s
            constant_lattice=constant_lattice,
            metadata=metadata,
            site_properties=site_properties,
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
            return Lattice(self.lattice)  # type: ignore

        latt = self.lattices[idx]  # type: ignore
        return Lattice(latt)

    @property
    def cumulative_displacements(self) -> np.ndarray:
        """Return cumulative displacement vectors from base positions.

        This differs from [displacements()][gemdat.trajectory.Trajectory.displacements]
        in that it ignores the periodic boundary conditions. Instead, it cumulatively
        tracks the lattice translation vector (jimage).
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

    def center_of_mass(self) -> Trajectory:
        """Return trajectory with center of mass for positions."""
        weights = []
        for s in self.species:
            assert isinstance(s, (Species, Element)), f'got {type(s)=}'
            weights.append(s.atomic_mass)

        positions_no_pbc = self.base_positions + self.cumulative_displacements

        center_of_mass = np.average(positions_no_pbc, axis=1, weights=weights).reshape(-1, 1, 3)

        return self.__class__(
            species=['X'],
            coords=center_of_mass,
            lattice=self.get_lattice(),
            metadata=self.metadata,
            time_step=self.time_step,
        )

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
            These species are assumed fixed, and are used to calculate drift
            (e.g. framework species).
        floating_species : None | str | Collection[str]
            These species are assumed floating, and is used to determine the
            fixed species.

        Returns
        -------
        drift : np.array
            Output array with average drift per frame.
        """
        if fixed_species:
            displacements = self.filter(species=fixed_species).displacements
        elif floating_species:
            species = set()
            for sp in self.species:
                assert isinstance(sp, Species), f'got {type(sp)=}'
                if sp.symbol not in floating_species:
                    species.add(sp)

            displacements = self.filter(species=species).displacements  # type: ignore
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
            These species are assumed fixed, and are used to calculate
            drift (e.g. framework species).
        floating_species : None | str | Collection[str]
            These species are assumed floating, and is used to determine
            the fixed species.

        Returns
        -------
        trajectory : Trajectory
            Ouput trajectory with positions corrected for drift
        """
        drift = self.drift(fixed_species=fixed_species, floating_species=floating_species)

        return self.__class__(
            species=self.species,
            coords=self.displacements - drift,
            lattice=self.get_lattice(),
            metadata=self.metadata,
            coords_are_displacement=True,
            base_positions=self.base_positions,
            time_step=self.time_step,
        )

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
        if isinstance(species, str):
            species = [species]

        idx = []
        for sp in self.species:
            assert isinstance(sp, (Species, Element))
            idx.append(sp.symbol in species)

        new_coords = self.positions[:, idx]
        new_species = list(compress(self.species, idx))

        return self.__class__(
            species=new_species,
            coords=new_coords,
            lattice=self.get_lattice(),
            metadata=self.metadata,
            time_step=self.time_step,
        )

    def split(self, n_parts: int = 10, equal_parts: bool = False) -> list[Trajectory]:
        """Split the trajectory in n similar parts.

        Parameters
        ----------
        n_parts : int
            n_parts
        equal_parts : bool = False
            Return equal parts, convenient for some routines

        Returns
        -------
        List[Trajectory]
        """
        interval = np.linspace(0, len(self) - 1, n_parts + 1, dtype=int)
        subtrajectories = [self[start:stop] for start, stop in pairwise(interval)]

        if equal_parts:
            minsize = len(self)

            # Get the smallest size
            for start, stop in pairwise(interval):
                size = stop - start
                minsize = min(minsize, size)

            # Trim all subtrajectories
            subtrajectories = [trajectory[0:minsize] for trajectory in subtrajectories]

        return subtrajectories

    def mean_squared_displacement(self) -> np.ndarray:
        """Computes the mean squared displacement using fast Fourier transform.

        The algorithm is described in [https://doi.org/10.1051/sfn/201112010].
        See also [https://stackoverflow.com/questions/34222272/
        computing-mean-square-displacement-using-python-and-fft].
        """
        r = self.cumulative_displacements
        lattice = self.get_lattice()
        r = lattice.get_cartesian_coords(r)

        pos = np.transpose(r, (1, 0, 2))
        n_times = pos.shape[1]

        # Autocorrelation term using FFT [https://doi.org/10.1051/sfn/201112010]:
        # - perform FFT, square it, and then perform inverse FFT
        fft_result = np.fft.ifft(np.abs(np.fft.fft(pos, n=2 * n_times, axis=-2)) ** 2, axis=-2)
        # - keep only the first n_times elements
        fft_result = fft_result[:, :n_times, :].real
        # - sum over the coordinates and divide by the corresponding time window
        S2 = np.sum(fft_result, axis=-1) / (n_times - np.arange(n_times)[None, :])

        # Compute the sum of squared displacements
        D = np.square(pos).sum(axis=-1)
        D = np.append(D, np.zeros((pos.shape[0], 1)), axis=-1)

        # Compute the first term of the MSD:
        # - compute 2* the sum of squared positions
        double_sum_D = 2 * np.sum(D, axis=-1)[:, None]
        # - compute the cumulative sum of the sum of squares of the positions
        cumsum_D = np.cumsum(
            np.insert(D[:, 0:-1], 0, 0, axis=-1) + np.flip(D, axis=-1), axis=-1
        )
        # - compute the first term in the MSD calculation
        S1 = (double_sum_D - cumsum_D)[:, :-1] / (n_times - np.arange(n_times)[None, :])

        msd = S1 - 2 * S2
        return msd

    def transitions_between_sites(
        self,
        sites: Structure,
        floating_specie: str,
        site_radius: float | dict[str, float] | None = None,
        site_inner_fraction: float = 1.0,
    ) -> Transitions:
        """Compute transitions between given sites for floating specie.

        Parameters
        ----------
        sites : pymatgen.core.structure.Structure
            Input structure with known sites
        floating_specie : str
            Name of the floating specie to calculate transitions for
        site_radius: Optional[float, dict[str, float]]
            A custom site radius in Ångstrom to determine
            if an atom is at a site. A dict keyed by the site label can
            be used to have a site per atom type, e.g.
            `site_radius = {'Li1': 1.0, 'Li2': 1.2}.
        site_inner_fraction:
            A fraction of the site radius which is determined to be the `inner site`
            which is used in jump calculations

        Returns
        -------
        transitions: Transitions
        """
        from gemdat.transitions import Transitions

        return Transitions.from_trajectory(
            trajectory=self,
            sites=sites,
            floating_specie=floating_specie,
            site_radius=site_radius,
            site_inner_fraction=site_inner_fraction,
        )

    def metrics(self) -> TrajectoryMetrics:
        """See [gemdat.TrajectoryMetrics][] for more info."""
        from .metrics import TrajectoryMetrics

        return TrajectoryMetrics(trajectory=self)

    @plot_backend
    def radial_distribution_between_species(self, *, module, **kwargs) -> RDFData:
        """See [gemdat.rdf.radial_distribution_between_species][] for more
        info."""
        from gemdat import rdf

        return rdf.radial_distribution_between_species(trajectory=self, **kwargs)

    @plot_backend
    def plot_displacement_per_atom(self, *, module, **kwargs):
        """See [gemdat.plots.displacement_per_atom][] for more info."""
        return module.displacement_per_atom(trajectory=self, **kwargs)

    @plot_backend
    def plot_displacement_per_element(self, *, module, **kwargs):
        """See [gemdat.plots.displacement_per_element][] for more info."""
        return module.displacement_per_element(trajectory=self, **kwargs)

    @plot_backend
    def plot_msd_per_element(self, *, module, **kwargs):
        """See [gemdat.plots.msd_per_element][] for more info."""
        return module.msd_per_element(trajectory=self, **kwargs)

    @plot_backend
    def plot_displacement_histogram(self, *, module, **kwargs):
        """See [gemdat.plots.displacement_histogram][] for more info."""
        return module.displacement_histogram(trajectory=self, **kwargs)

    @plot_backend
    def plot_frequency_vs_occurence(self, *, module, **kwargs):
        """See [gemdat.plots.frequency_vs_occurence][] for more info."""
        return module.frequency_vs_occurence(trajectory=self, **kwargs)

    @plot_backend
    def plot_vibrational_amplitudes(self, *, module, **kwargs):
        """See [gemdat.plots.vibrational_amplitudes][] for more info."""
        return module.vibrational_amplitudes(trajectory=self, **kwargs)
