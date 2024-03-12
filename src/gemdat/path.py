"""This module contains classes for computing optimal and percolating paths
between sites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import networkx as nx
import numpy as np
from pymatgen.core import Structure

from gemdat.volume import Volume

from .utils import nearest_structure_reference


@dataclass
class Pathway:
    """Container class for paths between sites.

    Attributes
    ----------
    sites: list[tuple]
        List of voxel coordinates of the sites defining the path
    energy: list[float]
        List of the energy along the path
    """

    sites: list[tuple[int, int, int]] | None = None
    energy: list[float] | None = None

    def __repr__(self):
        s = (
            f'Path: {self.start_site} -> {self.stop_site}',
            f'Steps: {len(self.sites)}',
            f'Total energy: {self.total_energy:.3f} eV',
        )

        return '\n'.join(s)

    @property
    def total_energy(self):
        """Return total energy for path."""
        return sum(self.energy)

    def cartesian_path(self,
                       volume: Volume) -> list[tuple[float, float, float]]:
        """Convert voxel coordinates to cartesian coordinates.

        Parameters
        ----------
        volume : Volume
            Volume object containing the grid information

        Returns
        -------
        cart_sites: list[tuple]
            List of cartesian coordinates of the sites defining the path
        """
        cart_sites = []
        if self.sites is None:
            raise ValueError('Voxel coordinates of the path are required.')
        for site in self.fractional_path(volume=volume):
            cartesian_coords = volume.lattice.get_cartesian_coords(site)
            cart_sites.append(tuple(cartesian_coords))
        return cart_sites

    def fractional_path(self,
                        volume: Volume) -> list[tuple[float, float, float]]:
        """Convert voxel coordinates to fractional coordinates.

        Parameters
        ----------
        volume : Volume
            Volume object containing the grid information

        Returns
        -------
        frac_sites: list[tuple]
            List of fractional coordinates of the sites defining the path
        """
        if self.sites is None:
            raise ValueError('Voxel coordinates of the path are required.')
        frac_sites = []
        for site in self.sites:
            fractional_coords = site / np.asarray(
                [x // volume.resolution for x in volume.lattice.lengths])
            frac_sites.append(tuple(fractional_coords))
        return frac_sites

    def wrap(self, dims: tuple[int, int, int]):
        """Wrap path in periodic boundary conditions in-place.

        Parameters
        ----------
        F: np.ndarray
            Grid in which the path sites will be wrapped
        """
        if self.sites is None:
            raise ValueError('Voxel coordinates of the path are required.')

        X, Y, Z = dims
        self.sites = [(x % X, y % Y, z % Z) for x, y, z in self.sites]

    def path_over_structure(
        self,
        structure: Structure,
        volume: Volume,
    ) -> tuple[list[str], list[np.ndarray]]:
        """Find the nearest site of the structure to the path sites.

        Parameters
        ----------
        structure : Structure
            Reference structure
        volume : Volume
            Volume object that contains the information about the nearest sites of the structure

        Returns
        -------
        nearest_structure_label: list[str]
            List of the label of the closest site of the reference structure
        nearest_structure_coord: list[np.ndarray]
            List of cartesian coordinates of the closest site of the reference structure
        """
        frac_sites = self.fractional_path(volume)
        nearest_structure_tree, nearest_structure_map = nearest_structure_reference(
            structure)

        # Get the indices of the nearest structure sites to the path sites
        nearest_structure_indices = [
            nearest_structure_tree.query(site)[1] for site in frac_sites
        ]
        # and use it to get its label and coordinates
        nearest_structure_label = [
            structure.labels[nearest_structure_map[index]]
            for index in nearest_structure_indices
        ]
        nearest_structure_coord = [
            structure.cart_coords[nearest_structure_map[index]]
            for index in nearest_structure_indices
        ]

        return nearest_structure_label, nearest_structure_coord

    @property
    def cost(self) -> float:
        """Calculate the path cost."""
        if self.energy is None:
            raise ValueError('Energy of the path is required.')
        return np.sum(self.energy)

    @property
    def start_site(self) -> tuple[int, int, int]:
        """Return first site."""
        if self.sites is None:
            raise ValueError('Voxel coordinates of the path are required.')
        return self.sites[0]

    @property
    def stop_site(self) -> tuple[int, int, int]:
        """Return stop site."""
        if self.sites is None:
            raise ValueError('Voxel coordinates of the path are required.')
        return self.sites[-1]


def free_energy_graph(F: np.ndarray | Volume,
                      max_energy_threshold: float = 1e20,
                      diagonal: bool = True) -> nx.Graph:
    """Compute the graph of the free energy for networkx functions.

    Parameters
    ----------
    F : np.ndarray | Volume
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
    movements = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                          (0, 0, 1), (0, 0, -1)])
    if diagonal:
        diagonal_movements = np.array([(1, 1, 0), (-1, -1, 0), (1, -1, 0),
                                       (-1, 1, 0), (1, 0, 1), (-1, 0, -1),
                                       (1, 0, -1), (-1, 0, 1), (0, 1, 1),
                                       (0, -1, -1), (0, 1, -1), (0, -1, 1),
                                       (1, 1, 1), (-1, -1, -1), (1, -1, -1),
                                       (-1, 1, 1)])
        movements = np.vstack((movements, diagonal_movements))

    G = nx.Graph()

    data = F.data if isinstance(F, Volume) else F

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

                G.add_edge(node,
                           neighbor,
                           weight=weight,
                           weight_exp=weight_exp)

    return G


_PATHFINDING_METHODS = Literal['dijkstra', 'bellman-ford', 'minmax-energy',
                               'dijkstra-exp', 'simple']


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


def _paths_too_similar(path: list, list_of_paths: list,
                       min_diff: float) -> bool:
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


def multiple_paths(
    *,
    F_graph: nx.Graph,
    start: tuple,
    stop: tuple,
    method: _PATHFINDING_METHODS = 'dijkstra',
    n_paths: int = 3,
    min_diff: float = 0.15,
) -> list[Pathway]:
    """ Calculate the Np shortest paths between two sites on the graph.
    This procedure is based the algorithm by Jin Y. Yen (https://doi.org/10.1287/mnsc.17.11.712)
    and its implementation in NetworkX. Only paths that are different by at least min_diff are considered.

    Parameters
    ----------
    F_graph : nx.Graph
        Graph of the free energy
    start : tuple
        Coordinates of the starting point
    stop: tuple
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

    # First compute the optimal path
    best_path = optimal_path(F_graph, start, stop, method)

    list_of_paths = [best_path]

    # Compute the iterator over all the short paths
    all_paths = nx.shortest_simple_paths(F_graph,
                                         source=start,
                                         target=stop,
                                         weight='weight')

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
    start: tuple,
    stop: tuple,
    method: _PATHFINDING_METHODS = 'dijkstra',
) -> Pathway:
    """Calculate the shortest cost-effective path using the desired method.

    Parameters
    ----------
    F_graph : nx.Graph
        Graph of the free energy
    start : tuple
        Coordinates of the starting point
    stop: tuple
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

    optimal_path = nx.shortest_path(F_graph,
                                    source=start,
                                    target=stop,
                                    weight=weight,
                                    method=method)

    if method == 'minmax-energy':
        optimal_path = _optimal_path_minmax_energy(F_graph, start, stop,
                                                   optimal_path)
    elif method not in ('dijkstra', 'bellman-ford', 'dijkstra-exp'):
        raise ValueError(f'Unknown method {method}')

    path_energy = [F_graph.nodes[node]['energy'] for node in optimal_path]
    path = Pathway(sites=optimal_path, energy=path_energy)
    return path


def _optimal_path_minmax_energy(
    F_graph: nx.Graph,
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
        minmax_energy = max(
            [F_graph.nodes[node]['energy'] for node in pruned_path])

        if minmax_energy < max_energy:
            optimal_path = pruned_path
            max_energy = minmax_energy

    return optimal_path


def find_best_perc_path(F: Volume,
                        peaks: np.ndarray,
                        percolate_x: bool = True,
                        percolate_y: bool = False,
                        percolate_z: bool = False) -> Pathway:
    """Calculate the best percolating path.

    Parameters
    ----------
    F : Volume
        Energy grid that will be used to calculate the shortest path
    peaks : np.ndarray
        List of the peaks that correspond to high probability regions
    percolate_x : bool
        If True, consider paths that percolate along the x dimension
    percolate_y : bool
        If True, consider paths that percolate along the y dimension
    percolate_z : bool
        If True, consider paths that percolate along the z dimension

    Returns
    -------
    best_percolating_path: Pathway
        Optimal path that percolates the graph in the specified directions
    """
    xyz_real = F.dims

    # Find percolation using virtual images along the required dimensions
    if not any([percolate_x, percolate_y, percolate_z]):
        print('Warning: percolation is not defined')
        return Pathway()

    # Tile the grind in the percolation directions
    F_data_periodic = np.tile(
        F.data, (1 + percolate_x, 1 + percolate_y, 1 + percolate_z))

    # Get F on a graph
    F_graph = free_energy_graph(F_data_periodic,
                                max_energy_threshold=1e7,
                                diagonal=True)

    # reaching the percolating image
    image = tuple(
        x * px
        for x, px in zip(xyz_real, (percolate_x, percolate_y, percolate_z)))

    # Find the lowest cost path that percolates along the x dimension
    best_cost = float('inf')
    best_path = Pathway()

    for start_point in peaks:

        # Get the stop point which is a periodic image of the peak
        stop_point = start_point + image

        # Find the shortest percolating path through this peak
        try:
            path = optimal_path(
                F_graph,
                tuple(start_point),
                tuple(stop_point),
            )
        except nx.NetworkXNoPath:
            continue

        cost = path.cost

        if cost < best_cost:
            best_cost = cost
            best_path = path

    # Before returning, wrap the path in the original volume
    best_path.wrap(F.dims)

    return best_path
