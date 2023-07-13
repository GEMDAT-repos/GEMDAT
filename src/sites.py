from __future__ import annotations

import warnings
from collections import Counter, defaultdict
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
from MDAnalysis.lib.pkdtree import PeriodicKDTree
from pymatgen.core import Lattice, Structure
from pymatgen.core.units import FloatWithUnit
from scipy.constants import Boltzmann, angstrom, elementary_charge

from .utils import bfill, ffill

if TYPE_CHECKING:
    from gemdat.data import SimulationData

NOSITE = -1


def lattice_is_similar(a: Lattice,
                       b: Lattice,
                       length_tol: float = 0.5,
                       angle_tol: float = 0.5) -> bool:
    """Return True if lattices are similar within given tolerance.

    Parameters
    ----------
    a, b : Lattice
        Input lattices
    length_tol : float, optional
        Length tolerance in Angstrom
    angle_tol : float, optional
        Angle tolerance in degrees

    Returns
    -------
    bool
        Return True if lattices are similar
    """
    for a_length, b_length in zip(a.lengths, b.lengths):
        if abs(a_length - b_length) > length_tol:
            return False

    for a_angle, b_angle in zip(a.angles, b.angles):
        if abs(a_angle - b_angle) > angle_tol:
            return False

    return True


class SitesData:

    def __init__(self, structure: Structure):
        self.structure = structure

    @property
    def site_coords(self):
        return self.structure.frac_coords

    @property
    def succes(self):
        """Alias for `self.transitions_parts`."""
        return self.transitions_parts

    def warn_if_lattice_not_similar(self, other_lattice: Lattice):
        this_lattice = self.structure.lattice

        if not lattice_is_similar(other_lattice, this_lattice):
            warnings.warn(f'Lattice mismatch: {this_lattice.parameters} '
                          f'vs. {other_lattice.parameters}')

    def calculate_all(self, data: SimulationData, extras: SimpleNamespace):
        """Calculate all parameters.

        Parameters
        ----------
        data : SimulationData
            Input simulation data
        extras : SimpleNamespace
            Extra parameters
        """
        self.warn_if_lattice_not_similar(data.structure.lattice)

        self.dist_close = self.calculate_dist_close(
            data, vibration_amplitude=extras.vibration_amplitude)
        self.atom_sites = self.calculate_atom_sites(
            data, diff_coords=extras.diff_coords)
        self.atom_sites_to, self.atom_sites_from = self.calculate_atom_sites_transitions(
        )

        self.all_transitions = self.calculate_transition_events()

        self.transitions = self.calculate_transitions_matrix(
            n_diffusing=extras.n_diffusing)
        self.transitions_parts = self.calculate_transitions_matrix_parts(
            n_steps=extras.n_steps,
            n_diffusing=extras.n_diffusing,
            n_parts=extras.n_parts)

        self.occupancy = self.calculate_occupancy()
        self.occupancy_parts = self.calculate_occupancy_parts(
            n_parts=extras.n_parts)

        self.sites_occupancy = self.calculate_site_occupancy(
            n_steps=extras.n_steps)
        self.sites_occupancy_parts = self.calculate_site_occupancy_parts(
            n_steps=extras.n_steps / extras.n_parts)

        self.atom_locations = self.sites_occupancy  # Is this correct? TODO: check
        self.atom_locations_parts = self.sites_occupancy_parts  # Is this correct? TODO: check

        self.jumps = self.calculate_jumps()
        self.jumps_parts = self.calculate_jumps_parts(n_steps=extras.n_steps,
                                                      n_parts=extras.n_parts)

        self.n_jumps = len(self.all_transitions)

        self.rates = self.calculate_rates(total_time=extras.total_time,
                                          n_parts=extras.n_parts,
                                          n_diffusing=extras.n_diffusing)
        self.activation_energies = self.calculate_activation_energies(
            total_time=extras.total_time,
            n_parts=extras.n_parts,
            n_diffusing=extras.n_diffusing,
            attempt_freq=extras.attempt_freq,
            temperature=data.temperature)

        self.jump_diffusivity = self.calculate_jump_diffusivity(
            lattice=data.lattice,
            n_diffusing=extras.n_diffusing,
            total_time=extras.total_time,
            dimensions=extras.diffusion_dimensions)
        self.correlation_factor = extras.tracer_diff / self.jump_diffusivity

        self.collective, self.coll_jumps, self.n_solo_jumps = self.calculate_collective(
            lattice=data.lattice,
            attempt_freq=extras.attempt_freq,
            time_step=data.time_step)
        self.solo_frac = self.n_solo_jumps / len(self.all_transitions)
        self.coll_count = len(self.collective)

    def calculate_dist_close(self, data: SimulationData,
                             vibration_amplitude: float):
        """Calculate tolerance wihin which atoms are considered to be close to
        a site.

        Parameters
        ----------
        data : SimulationData
            Simulation data
        vibration_amplitude : float
            Vibration amplitude

        Returns
        -------
        dist_close : float
            Atoms within this distance (in Angstrom) are considered to be close to a site
        """
        dist_close = 2 * vibration_amplitude

        pdist = data.lattice.get_all_distances(self.site_coords,
                                               self.site_coords)
        min_dist = np.min(pdist[np.triu_indices_from(pdist, k=1)])

        if min_dist < 2 * dist_close:
            # Crystallographic sites are overlapping with the chosen dist_close, making it smaller
            dist_close = (0.5 * min_dist) - 0.005

            # Two crystallographic sites are within half an Angstrom of each other
            # This is NOT realistic, check/change the given crystallographic site
            if dist_close * 2 < 0.5:
                idx = np.argwhere(pdist == min_dist)

                lines = []

                for i, j in idx:
                    self.structure.sites[i]
                    site_j = self.structure.sites[j]
                    lines.append('\nToo close:')
                    lines.append(
                        '{site_i.specie.name}({i}) {site_i.frac_coords} - ')
                    lines.append(
                        f'{site_j.specie.name}({j}) {site_j.frac_coords}')

                msg = ''.join(lines)

                raise ValueError(
                    f'Crystallographic sites are too close together (expected: >{dist_close*2:.4f}, '
                    f'got: {min_dist:.4f} for {msg}')

        return dist_close

    def calculate_atom_sites(self, data: SimulationData,
                             diff_coords: np.ndarray) -> np.ndarray:
        """Calculate nearest site for each atom coordinate.

        Note: This is a slow operation, because a pairwise distance matrix between all `coords` and
        all `site_coords` has to be generated. This includes lattice translations. The nearest site
        may be in the neighbouring unit cell.

        Parameters
        ----------
        data : SimulationData
            Simulation data
        diff_coords:
            Input array with (diffusing) atom coordinates [time, atom, (x, y, z)]

        Returns
        -------
        atom_sites : np.ndarray
            Output array with site locations for each atom at each time step [time, atom].
            The value corresponds to the index in the `site_coords`.
            -1 indicates that atom is not at any site.
        """
        coords = diff_coords

        # Unit cell parameters
        lattice = data.lattice

        # Atoms within this distance (in Angstrom) are considered to be close to a site
        dist_close = self.dist_close

        # fractional coordinates

        # Input array with site coordinates [site, (x, y, z)]
        site_cart_coords = np.dot(self.site_coords, lattice.matrix)
        site_coords_tree: PeriodicKDTree = PeriodicKDTree(
            box=np.array(lattice.parameters, dtype=np.float32))
        site_coords_tree.set_coords(site_cart_coords, cutoff=dist_close)

        atom_sites = []

        for atom_index, atom_coords in enumerate(coords.swapaxes(0, 1)):

            # index and distance of nearest site
            atom_cart_coords = np.dot(atom_coords, lattice.matrix)
            site_index = site_coords_tree.search_tree(atom_cart_coords,
                                                      dist_close)

            # construct mapping
            atom_site = np.full((atom_coords.shape[0], 1), NOSITE)
            for index, site in site_index:
                atom_site[index] = site

            atom_sites.append(atom_site)

        return np.hstack(atom_sites)

    def calculate_atom_sites_transitions(
            self) -> tuple[np.ndarray, np.ndarray]:
        """Calculate atom transition states per time step by back/forward filling
        `self.atom_sites`.

        Returns
        -------
        atom_sites_from, atom_sites_to : tuple[np.ndarray, np.ndarray]
            Output arrays with atom transition states. `atom_sites_from` contains
            the index of the previous site for every atom, and `atom_sites_to` the
            next site. If `atom_sites_from` and `atom_sites_to` are equal, the atom
            is currently sitting at that site.
        """
        atom_sites_from = ffill(self.atom_sites, fill_val=NOSITE, axis=0)
        atom_sites_to = bfill(self.atom_sites, fill_val=NOSITE, axis=0)

        return atom_sites_from, atom_sites_to

    def calculate_occupancy(self):
        """Calculate occupancy per site.

        Returns
        -------
        occupancy: dict[int, int]
            For each site, count for how many time steps it is occupied by an atom
        """
        return _calculate_occupancy(self.atom_sites)

    def calculate_occupancy_parts(self, n_parts: int) -> list[dict[int, int]]:
        """Calculate occupancy per site, divided in parts.

        Parameters
        ----------
        n_parts : int, optional
            Number of parts to split into

        Returns
        -------
        occupancy_parts: list[dict[int, int]]
            Returns a list of dicts, where each dict corresponds to a part of the data.
            Eech dict counts the number of time steps a site is occupied by an atom
        """
        split_atom_sites = np.split(self.atom_sites, n_parts)
        return [_calculate_occupancy(part) for part in split_atom_sites]

    def calculate_site_occupancy(self, *, n_steps: int) -> dict[str, float]:
        """Calculate percentage occupancy per unique site.

        Parameters
        ----------
        n_steps : int
            Number of steps in time series

        Returns
        -------
        site_occopancy : dict[str, float]
            Percentage occupancy per unique site
        """
        labels = self.structure.labels
        return _calculate_site_occupancy(occupancy=self.occupancy,
                                         labels=labels,
                                         n_steps=n_steps)

    def calculate_site_occupancy_parts(self, *,
                                       n_steps: int) -> list[dict[str, float]]:
        """Calculate percentage occupancy per unique site per part.

        Parameters
        ----------
        n_steps : int
            Number of steps in time series

        Returns
        -------
        site_occopancy : list[dict[str, float]]
            Return a list of dicts, where each dict contains the percentage occupancy per unique site
        """
        labels = self.structure.labels
        return [
            _calculate_site_occupancy(occupancy=occupancy,
                                      labels=labels,
                                      n_steps=n_steps)
            for occupancy in self.occupancy_parts
        ]

    def calculate_transition_events(self):
        """Find transitions between sites.

        Returns
        -------
        all_transitions : np.ndarray
            Output array with transition events.
            Contains 5 columns: atom index, time start, time stop, site start, site stop
        """
        return _calculate_transition_events(atom_sites=self.atom_sites)

    def calculate_transitions_matrix(self, n_diffusing: int):
        """Convert list of transition events to dense transitions matrix.

        Parameters
        ----------
        n_diffusing : int
            Number of diffusing elements. This defines the shape of the output matrix.

        Returns
        -------
        transitions_matrix : np.ndarray
            Square matrix with number of each transitions
        """
        return _calculate_transitions_matrix(self.all_transitions,
                                             n_diffusing=n_diffusing)

    def calculate_transitions_matrix_parts(self, *, n_steps: int,
                                           n_diffusing: int,
                                           n_parts: int) -> np.ndarray:
        """Divide list of transition events in equal parts and convert to dense
        transition matrices.

        Note: equivalent to `sites.succes` in the matlab code.

        Parameters
        ----------
        n_steps : int
            Number of steps
        n_diffusing : int
            Number of diffusing elements. This defines the shape of the output matrix.
        n_parts : int
            Number of parts to divide the transitions events list into

        Returns
        -------
        transitions_matrix_parts : np.ndarray
            Stacked square matrices with number of each transitions.
            The number of stacked matrices is equal to the number of parts specified.
        """
        split_transitions = _split_transitions_in_parts(
            self.all_transitions, n_steps, n_parts)
        return np.stack([
            _calculate_transitions_matrix(part, n_diffusing=n_diffusing)
            for part in split_transitions
        ])

    def calculate_jumps(self) -> dict[tuple[str, str], int]:
        """Calculate number of jumpst between sites.

        Returns
        -------
        jumps : dict[tuple[str, str], int]
            Dictionary with number of jumpst per sites combination
        """
        labels = self.structure.labels
        defaultdict(list)

        jumps = Counter([(labels[i], labels[j])
                         for i, j in self.all_transitions[:, 1:3]])

        return jumps

    def calculate_jumps_parts(
            self, *, n_steps: int,
            n_parts: int) -> dict[tuple[str, str], list[int]]:
        """Calculate number of jumpst between sites divided in parts.

        Parameters
        ----------
        n_steps : int
            Number of steps
        n_parts : int
            Number of parts to divide the transitions events list into

        Returns
        -------
        jumps : dict[tuple[str, str], list[int]]
            Dictionary with number of jumpst per sites combination
        """
        labels = self.structure.labels
        all_transitions_parts = _split_transitions_in_parts(
            self.all_transitions, n_steps, n_parts)

        jumps_parts = defaultdict(list)

        for part in all_transitions_parts:
            c = Counter([(labels[i], labels[j]) for i, j in part[:, 1:3]])
            for k, v in c.items():
                jumps_parts[k].append(v)

        return jumps_parts

    def calculate_rates(
            self, *, total_time: int, n_parts: int,
            n_diffusing: int) -> dict[tuple[str, str], tuple[float, float]]:
        """Calculate jump rates (total jumps / second).

        Parameters
        ----------
        total_time : int
            Total time for the simulation
        n_parts : int, optional
            Number of parts to split into
        n_diffusing : int
            Number of diffusing atoms

        Returns
        -------
        rates : dict[tuple[str, str], tuple[float, float]]
            Dictionary with jump rates and standard deviations between site pairs
        """
        rates = {}

        for site_pair, n_jumps in self.jumps_parts.items():
            part_time = total_time / n_parts
            denom = n_diffusing * part_time

            jump_freq_mean = np.mean(n_jumps) / denom
            jump_freq_std = np.std(n_jumps, ddof=1) / denom

            rates[site_pair] = jump_freq_mean, jump_freq_std

        return rates

    def calculate_activation_energies(
            self, *, total_time: int, n_parts: int, n_diffusing: int,
            attempt_freq: float,
            temperature: float) -> dict[tuple[str, str], tuple[float, float]]:
        """Calculate activation energies for jumps (UNITS?).

        Parameters
        ----------
        total_time : int
            Total time for the simulation
        n_parts : int
            Number of parts to split into
        n_diffusing : int
            Number of diffusing atoms
        attempt_freq : float
            Jump attempt frequency
        temperature : float
            Temperature of the simulation

        Returns
        -------
        e_act : dict[tuple[str, str], tuple[float, float]]
            Dictionary with jump activation energies and standard deviations between site pairs.
        """
        e_act = {}

        for i, ((site_start, site_stop),
                n_jumps) in enumerate(self.jumps_parts.items()):
            n_jumps_arr = np.array(n_jumps)

            part_time = total_time / n_parts

            atom_percentage = self.atom_locations_parts[i][site_start]

            denom = atom_percentage * n_diffusing * part_time

            eff_rate = n_jumps_arr / denom

            # For A-A jumps divide by two for a fair comparison of A-A jumps vs. A-B and B-A
            if site_start == site_stop:
                eff_rate /= 2

            e_act_arr = -np.log(eff_rate / attempt_freq) * (
                Boltzmann * temperature) / elementary_charge

            e_act[site_start,
                  site_stop] = np.mean(e_act_arr), np.std(e_act_arr, ddof=1)

        return e_act

    def calculate_jump_diffusivity(self, lattice: Lattice, n_diffusing: int,
                                   total_time: float,
                                   dimensions: int) -> float:
        """Calculate jump diffusivity.

        Parameters
        ----------
        lattice : Lattice
            Lattice of the simulation data
        n_diffusing : int
            Number of diffusing elements
        total_time : float
            Total simulation time
        dimensions : int
            Number of diffusion dimensions

        Returns
        -------
        jump_diff : float
            Jump diffusivity in m^2/s
        """
        structure = self.structure

        pdist = lattice.get_all_distances(structure.frac_coords,
                                          structure.frac_coords)

        jump_diff = np.sum(pdist**2 * self.transitions)
        jump_diff *= angstrom**2 / (2 * dimensions * n_diffusing * total_time)

        jump_diff = FloatWithUnit(jump_diff, 'm^2 s^-1')

        print(f'{jump_diff=} {jump_diff.unit}')

        return jump_diff

    def calculate_collective(
            self,
            lattice: Lattice,
            attempt_freq: float,
            time_step: float,
            dist_collective: float = 4.5) -> tuple[list, list, int]:
        """Calculate collective jumps.

        Parameters
        ----------
        lattice : Lattice
            Lattice for distance calculations
        attempt_freq : float
            Attempt frequency in ...
        time_step : float
            Time step in seconds
        dist_collective : float, optional
            Maximum distance for collective motions in Angstrom

        Returns
        -------
        tuple[list, list, int]
            collective, list of indices to collective jump events
            coll_jumps, list with start/stop site indices involved in collective jumps
            n_solo_jumps, number of solo jumps
        """
        from math import ceil
        coll_steps = ceil(1.0 / (attempt_freq * time_step))

        time_col = 4
        events = self.all_transitions
        event_order = events[:, time_col].argsort()

        coll_jumps = []
        collective = []

        n_solo_jumps = 0

        structure = self.structure

        for i, event_i in enumerate(event_order[:-1]):

            for j, event_j in enumerate(event_order[i + 1:], start=1):

                atom_i, start_site_i, stop_site_i, _, stop_time_i = events[
                    event_i]
                atom_j, start_site_j, stop_site_j, _, stop_time_j = events[
                    event_j]

                if stop_time_j - stop_time_i > coll_steps:
                    break

                if atom_i == atom_j:
                    n_solo_jumps += 1
                    continue

                a = structure.frac_coords[[start_site_i, stop_site_i]]
                b = structure.frac_coords[[start_site_j, stop_site_j]]

                dists = lattice.get_all_distances(a, b)

                if np.any(dists < dist_collective):
                    collective.append((event_i, event_j))
                    coll_jumps.append(((start_site_i, stop_site_i),
                                       (start_site_j, stop_site_j)))

                else:
                    n_solo_jumps += 1

        return collective, coll_jumps, n_solo_jumps


def _calculate_transition_events(*, atom_sites: np.ndarray) -> np.ndarray:
    """Find transitions between sites.

    Parameters
    ----------
    atom_sites : np.ndarray
        Input array with atom sites

    Returns
    -------
    all_transitions : np.ndarray
        Output array with transition events.
        Contains 5 columns: atom index, time start, time stop, site start, site stop
    """
    all_transitions = []

    for atom_index, atom_site in enumerate(atom_sites.T):

        # Indices when atom jumps to new or back to same site
        i, = np.nonzero((atom_site != np.roll(atom_site, shift=1))
                        & (atom_site >= 0))

        # Log transition events
        i_event = np.nonzero(atom_site[i] != np.roll(atom_site[i], shift=-1))
        time_start = i[i_event]
        time_stop = np.roll(i, shift=-1)[i_event]
        transitions = np.vstack([
            np.ones_like(time_start) * atom_index,
            atom_site[time_start],
            atom_site[time_stop],
            time_start,
            time_stop,
        ]).T

        # Drop last event (side effect of np.roll)
        transitions = transitions[:-1]
        all_transitions.append(transitions)

    return np.vstack(all_transitions)


def _calculate_transitions_matrix(all_transitions: np.ndarray,
                                  n_diffusing: int) -> np.ndarray:
    """Convert list of transition events to dense transitions matrix.

    Parameters
    ----------
    all_transitions : np.ndarray
        Input array with transition events
    n_diffusing : int
        Number of diffusing elements. This defines the shape of the output matrix.

    Returns
    -------
    np.ndarray
        Square matrix with number of each transitions
    """
    start_col = 1  # transition starts
    stop_col = 2  # transition stop

    transitions = np.zeros((n_diffusing, n_diffusing), dtype=int)
    idx, counts = np.unique(all_transitions[:, [start_col, stop_col]],
                            return_counts=True,
                            axis=0)
    start_idx, stop_idx = idx.T
    transitions[start_idx, stop_idx] = counts
    return transitions


def _split_transitions_in_parts(all_transitions: np.ndarray,
                                n_steps: int,
                                n_parts=10) -> list[np.ndarray]:
    """Split list of transition events into equal parts in time.

    Parameters
    ----------
    all_transitions : np.ndarray
        Input array with transition events
    n_steps : int
        Number of time steps
    n_parts : int, optional
        Number of parts to split into

    Returns
    -------
    transitions_parts : np.ndarray
        Sorted list of transition events split into equal parts.
        The first dimension corresponds to `n_parts`.
    """
    col = 4

    bins = np.linspace(0, n_steps + 1, n_parts + 1, dtype=int)
    parts = np.digitize(all_transitions[:, col], bins=bins)
    parts = parts[parts.argsort()]
    splits = np.unique(parts, return_index=True)[1][1:]

    sorted_transitions = all_transitions[all_transitions[:, col].argsort()]

    return np.split(sorted_transitions, splits)


def _calculate_occupancy(atom_sites: np.ndarray) -> dict[int, int]:
    """Calculate occupancy per site.

    Parameters
    ----------
    atom_sites : np.ndarray
        Input array with atom sites

    Returns
    -------
    occupancy : dict[int, int]
        For each site, count for how many time steps it is occupied by an atom
    """
    unq, counts = np.unique(atom_sites, return_counts=True)
    occupancy = dict(zip(unq, counts))
    occupancy.pop(NOSITE, None)
    return occupancy


def _calculate_site_occupancy(*, occupancy: dict[int, int], labels: list[str],
                              n_steps: int) -> dict[str, float]:
    """Calculate percentage occupancy per unique site.

    Parameters
    ----------
    occupancy : dict[int, int]
        Occupancy dict
    labels : list[str]
        Site labels
    n_steps : int
        Number of steps in time series

    Returns
    -------
    dict[str, float]
        Percentage occupancy per unique site
    """
    counts = defaultdict(list)

    assert all(v >= 0 for v in occupancy)

    for k, v in occupancy.items():
        label = labels[k]
        counts[label].append(v)

    div = len(labels) * n_steps
    site_occupancies = {k: sum(v) / div for k, v in counts.items()}
    # site_occupancies['total'] = sum(chain(*counts.values())) / (len(occupancy) * n_steps)

    return site_occupancies
