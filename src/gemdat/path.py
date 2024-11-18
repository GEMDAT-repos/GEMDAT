"""This module contains classes for computing optimal and percolating paths
between sites."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Literal

import networkx as nx
import numpy as np
from pymatgen.core import Structure
from pymatgen.core.units import FloatWithUnit

from gemdat.volume import FreeEnergyVolume

from ._plot_backend import plot_backend
from .utils import nearest_structure_reference

if TYPE_CHECKING:
    from pymatgen.core import Lattice, PeriodicSite


@dataclass
class Pathway:
    """Container class for paths between sites.

    Attributes
    ----------
    sites: list[tuple]
        List of voxel coordinates of the sites defining the path
    energy: list[float]
        List of the energy along the path
    dims: [int, int, int] | None
        Voxel dimensions of bounding box. If set (usually to `Volume.dims`),
        enable some site transformations.
    """

    sites: list[tuple[int, int, int]]
    energy: list[float]
    dims: tuple[int, int, int] | None = None

    def __repr__(self):
        s = (
            f'Path: {self.start_site} -> {self.stop_site}',
            f'Steps: {len(self.sites)}',
            f'Total energy: {self.total_energy:.3f} eV',
            f'Dimensions: {self.dims}',
        )

        return '\n'.join(s)

    @property
    def total_energy(self):
        """Return total energy for path."""
        return sum(self.energy)

    def total_length(self, lattice: Lattice) -> FloatWithUnit:
        """Return total length of pathway in Ångstrom.

        Parameters
        ----------
        lattice : Lattice
            Lattice parameters

        Returns
        -------
        length : FloatWithUnit
            Total distance in Ångstrom
        """
        length = 0.0
        for a, b in pairwise(self.frac_sites()):
            dist, _ = lattice.get_distance_and_image(a, b)
            assert dist
            length += dist
        return FloatWithUnit(length, 'ang')

    def wrapped_sites(self) -> list[tuple[int, int, int]]:
        """Wrap sites to bounding box.

        Returns
        -------
        np.ndarray
            Voxel coordinates wrapped to bounding box.
        """
        if not self.dims:
            raise AttributeError(f'Dimensions are needed for this method {self.dims=}')
        xdim, ydim, zdim = self.dims
        return [(x % xdim, y % xdim, z % xdim) for x, y, z in self.sites]

    def frac_sites(self) -> np.ndarray:
        """Return fractional site coordinates.

        Note that these wrap around the periodic boundary conditions.

        Returns
        -------
        np.ndarray
            Fractional coordinates for sites
        """
        if not self.dims:
            raise AttributeError(f'Dimensions are needed for this method {self.dims=}')
        sites = self.wrapped_sites()
        return (np.array(sites) + 0.5) / np.array(self.dims)

    def path_over_structure(
        self,
        structure: Structure,
    ) -> list[PeriodicSite]:
        """Find the nearest site of the structure to the path sites.

        Parameters
        ----------
        structure : Structure
            Reference structure

        Returns
        -------
        nearest_sites: list[PeriodicSite]
            List closest sites of the reference structure
        """
        frac_sites = np.array(self.frac_sites())

        nearest_structure_tree, nearest_structure_map = nearest_structure_reference(structure)

        # Get the indices of the nearest structure sites to the path sites
        nearest_structure_indices = [
            nearest_structure_tree.query(site)[1] for site in frac_sites
        ]
        # and use it to get its label and coordinates
        nearest_sites = [
            structure[nearest_structure_map[index]] for index in nearest_structure_indices
        ]
        return nearest_sites

    @property
    def start_site(self) -> tuple[int, int, int]:
        """Return first site."""
        return self.sites[0]

    @property
    def stop_site(self) -> tuple[int, int, int]:
        """Return stop site."""
        return self.sites[-1]

    @plot_backend
    def plot_energy_along_path(self, module, **kwargs):
        """See [gemdat.plots.energy_along_path][] for more info."""
        return module.energy_along_path(path=self, **kwargs)

    @plot_backend
    def plot_path_on_grid(self, module, **kwargs):
        """See [gemdat.plots.path_on_grid][] for more info."""
        return module.path_on_grid(path=self, **kwargs)


def free_energy_graph(
    F: np.ndarray | FreeEnergyVolume,
    max_energy_threshold: float = 1e20,
    diagonal: bool = True,
) -> nx.Graph:
    """Compute the graph of the free energy for networkx functions.

    Parameters
    ----------
    F : np.ndarray | FreeEnergyVolume
        Free energy on the 3d grid
    max_energy_threshold : float, optional
        Maximum energy threshold for the path to be considered valid
    diagonal : bool
        If True, allows diagonal grid moves

    Returns
    -------
    G : nx.Graph
        Graph of free energy
    """

    # Define possible movements in 3D space
    movements = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])
    if diagonal:
        diagonal_movements = np.array(
            [
                (1, 1, 0),
                (-1, -1, 0),
                (1, -1, 0),
                (-1, 1, 0),
                (1, 0, 1),
                (-1, 0, -1),
                (1, 0, -1),
                (-1, 0, 1),
                (0, 1, 1),
                (0, -1, -1),
                (0, 1, -1),
                (0, -1, 1),
                (1, 1, 1),
                (-1, -1, -1),
                (1, -1, -1),
                (-1, 1, 1),
            ]
        )
        movements = np.vstack((movements, diagonal_movements))

    G = nx.Graph()

    data = F.data if isinstance(F, FreeEnergyVolume) else F

    for index, Fi in np.ndenumerate(data):
        if 0 <= Fi < max_energy_threshold:
            G.add_node(index, energy=Fi)

    for node in G.nodes:
        for move in movements:
            neighbor = tuple((node + move) % data.shape)
            if neighbor in G.nodes:
                weight = 0.5 * (data[node] + data[neighbor])
                exp_n_energy = np.exp(weight)
                if exp_n_energy < max_energy_threshold:
                    weight_exp = exp_n_energy
                else:
                    weight_exp = max_energy_threshold

                G.add_edge(node, neighbor, weight=weight, weight_exp=weight_exp)

    return G


_PATHFINDING_METHODS = Literal[
    'dijkstra', 'bellman-ford', 'minmax-energy', 'dijkstra-exp', 'simple'
]


def calculate_path_difference(path1: list, path2: list) -> float:
    """Calculate the difference between two paths. This difference is defined
    as the percentage of sites that are not shared between the two paths.

    Parameters
    ----------
    path1 : list
        List of sites defining the first path
    path2 : list
        List of sites defining the second path

    Returns
    -------
    difference : float
        Difference between the two paths
    """

    # Find the shortest and longest paths
    shortest, longest = sorted((path1, path2), key=len)

    # Calculate the number of nodes shared between the shortest and longest paths
    shared_nodes = 0
    for node in shortest:
        if node in longest:
            shared_nodes += 1

    return 1 - (shared_nodes / len(shortest))


def _paths_too_similar(path: list, list_of_paths: list, min_diff: float) -> bool:
    """Check if the path is too similar to the other paths.

    Parameters
    ----------
    path : list
        List of sites defining the path
    list_of_paths : list
        List of Pathway objects defining the other paths
    min_diff : float
        Minimum difference between the paths

    Returns
    -------
    too_similar : bool
        True if the path is too similar to the other paths
    """

    for good_path in list_of_paths:
        if calculate_path_difference(path, good_path.sites) < min_diff:
            return True
    return False


def optimal_n_paths(
    F_graph: nx.Graph,
    *,
    start: Collection,
    stop: Collection,
    method: _PATHFINDING_METHODS = 'dijkstra',
    n_paths: int = 3,
    min_diff: float = 0.15,
) -> list[Pathway]:
    """Calculate the n_paths shortest paths between two sites on the graph.
    This procedure is based the algorithm by Jin Y. Yen
    (https://doi.org/10.1287/mnsc.17.11.712) and its implementation in NetworkX.
    Only paths that are different by at least min_diff are considered.

    .. warning::
        Notice that this function in based on networkx.all_shortest_paths, which tends
        to identify first small variations of the optimal path. A custom graph pruning
        approach is suggested to accommodate different needs.

    Parameters
    ----------
    F_graph : nx.Graph
        Graph of the free energy
    start : Collection
        Coordinates of the starting point
    stop: Collection
        Coordinates of the stopping point
    method : str
        Method used to calculate the shortest path. Options are:
        - 'simple': Shortest, unweighted path
        - 'dijkstra': Dijkstra's algorithm
        - 'bellman-ford': Bellman-Ford algorithm
        - 'minmax-energy': Minmax energy algorithm
        - 'dijkstra-exp': Dijkstra's algorithm with exponential weights
    Npaths : int
        Number of paths to be calculated
    min_diff : float
        Minimum difference between the paths

    Returns
    -------
    list_of_paths: list[Pathway]
        List of the n_paths shortest paths between the start and stop sites
    """
    start = tuple(start)
    stop = tuple(stop)

    # First compute the optimal path
    best_path = optimal_path(F_graph, start=start, stop=stop, method=method)

    list_of_paths = [best_path]

    # Compute the iterator over all the short paths
    all_paths = nx.shortest_simple_paths(F_graph, source=start, target=stop, weight='weight')

    # Attempt to find the Np shortest paths
    for idx, path in enumerate(all_paths):
        if _paths_too_similar(path, list_of_paths, min_diff):
            continue

        path_energy = [F_graph.nodes[node]['energy'] for node in path]
        list_of_paths.append(Pathway(sites=path, energy=path_energy))

        if len(list_of_paths) == n_paths:
            break

    return list_of_paths


def optimal_path(
    F_graph: nx.Graph,
    *,
    start: Collection,
    stop: Collection,
    method: _PATHFINDING_METHODS = 'dijkstra',
) -> Pathway:
    """Calculate the shortest cost-effective path using the desired method.

    Parameters
    ----------
    F_graph : nx.Graph
        Graph of the free energy
    start : Collection
        Coordinates of the starting point
    stop: Collection
        Coordinates of the stoping point
    method : str
        Method used to calculate the shortest path. Options are:
        - 'simple': Shortest, unweighted path
        - 'dijkstra': Dijkstra's algorithm
        - 'bellman-ford': Bellman-Ford algorithm
        - 'minmax-energy': Minmax energy algorithm
        - 'dijkstra-exp': Dijkstra's algorithm with exponential weights

    Returns
    -------
    path: Pathway
        Optimal path on the graph between start and stop
    """
    if method == 'simple':
        weight = None
    elif method == 'dijkstra-exp':
        weight = 'weight_exp'
    else:
        weight = 'weight'

    if method in ('dijkstra-exp', 'minmax-energy', 'simple'):
        method = 'dijkstra'

    start = tuple(start)
    stop = tuple(stop)

    optimal_path = nx.shortest_path(
        F_graph, source=start, target=stop, weight=weight, method=method
    )

    if method == 'minmax-energy':
        optimal_path = _optimal_path_minmax_energy(
            F_graph, start=start, stop=stop, optimal_path=optimal_path
        )
    elif method not in ('dijkstra', 'bellman-ford', 'dijkstra-exp'):
        raise ValueError(f'Unknown method {method}')

    path_energy = [F_graph.nodes[node]['energy'] for node in optimal_path]
    path = Pathway(sites=optimal_path, energy=path_energy)
    return path


def _optimal_path_minmax_energy(
    F_graph: nx.Graph,
    *,
    start: tuple[int, int, int],
    stop: tuple[int, int, int],
    optimal_path: list,
) -> list:
    """Find the optimal path that has the minimum maximum-energy.

    Parameters
    ----------
    F_graph : nx.Graph
        Graph of the free energy
    start : tuple
        Coordinates of the starting point
    stop: tuple
        Coordinates of the stoping point
    optimal_path : list
        List of the nodes of the optimal path

    Returns
    -------
    optimal_path: list
        Optimal path on the graph between start and stop
    """

    max_energy = max([F_graph.nodes[node]['energy'] for node in optimal_path])
    minmax_energy = max_energy
    pruned_F_graph = F_graph.copy()

    while minmax_energy <= max_energy:
        # Find the node of the path with the highest energy
        max_node = max(optimal_path, key=lambda x: F_graph.nodes[x]['energy'])
        # remove this node from the graph
        pruned_F_graph.remove_node(max_node)
        # recompute the path
        pruned_path = nx.shortest_path(
            pruned_F_graph,
            source=start,
            target=stop,
            weight='weight',
        )
        minmax_energy = max([F_graph.nodes[node]['energy'] for node in pruned_path])

        if minmax_energy < max_energy:
            optimal_path = pruned_path
            max_energy = minmax_energy

    return optimal_path


def optimal_percolating_path(
    F: FreeEnergyVolume,
    *,
    peaks: np.ndarray,
    percolate: str,
) -> Pathway | None:
    """Calculate the optimal percolating path.

    Parameters
    ----------
    F : FreeEnergyVolume
        Energy grid that will be used to calculate the shortest path
    peaks : np.ndarray
        List of the peaks that correspond to high probability regions
    percolate : str
        Directions to percolate, e.g. 'x' to consider paths that
        percolate along the x dimension, 'yz' for the y/z dimension,
        or any other combinition of 'x', 'y', and 'z'.

    Returns
    -------
    best_percolating_path: Pathway
        Optimal path that percolates the graph in the specified directions
    """
    percolate_xyz = np.array([dim in percolate for dim in 'xyz'])

    if not percolate_xyz.any():
        raise ValueError('percolation is not defined')

    # Tile the grind in the percolation directions
    F_data_periodic = np.tile(F.data, tuple(1 + percolate_xyz))

    # Get F on a graph
    F_graph = free_energy_graph(F_data_periodic, max_energy_threshold=1e7)

    # reaching the percolating image
    image = F.dims * percolate_xyz

    # Find the lowest cost path that percolates along the x dimension
    best_cost = float('inf')
    best_path = None

    for start_point in peaks:
        # Get the stop point which is a periodic image of the peak
        stop_point = start_point + image

        # Find the shortest percolating path through this peak
        try:
            path = optimal_path(
                F_graph,
                start=start_point,
                stop=stop_point,
            )
        except nx.NetworkXNoPath:
            continue

        cost = path.total_energy

        if cost < best_cost:
            best_cost = cost
            best_path = path

    if best_path:
        # Before returning, set dimensions of original volume
        best_path.dims = F.dims

    return best_path
