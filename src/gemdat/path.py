"""This module contains classes for computing optimal and percolating paths
between sites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Tuple

import networkx as nx
import numpy as np
from pymatgen.core import Structure

from gemdat.volume import Volume


@dataclass
class Pathway:
    """Container class for paths between sites.

    Attributes
    ----------
    sites: list[tuple]
        List of voxel coordinates of the sites defining the path
    cart_sites: list[tuple]
        List of cartesian coordinates of the sites defining the path
    frac_sites: list[tuple]
        List of fractional coordinates of the sites defining the path
    energy: list[float]
        List of the energy along the path
    nearest_structure_coord: list[np.ndarray]
        List of cartesian coordinates of the closest site of the reference structure
    nearest_structure_label: list[str]
        List of the label of the closest site of the reference structure
    """

    sites: List[Tuple[int, int, int]] | None = None
    cart_sites: List[Tuple[float, float, float]] | None = None
    frac_sites: List[Tuple[float, float, float]] | None = None
    energy: list[float] | None = None
    nearest_structure_coord: list[np.ndarray] | None = None
    nearest_structure_label: list[str] | None = None

    def cartesian_path(self, vol: Volume):
        """Convert voxel coordinates to cartesian coordinates.

        Parameters
        ----------
        vol : Volume
            Volume object containing the grid information
        """
        # Manually get the fractional coordinates of the sites, then use
        # lattice.get_cartesian_coords and make into tuple
        self.cart_sites = []
        if self.sites is None:
            raise ValueError('Voxel coordinates of the path are required.')
        for site in self.sites:
            fractional_coords = site / np.asarray(
                [x // vol.resolution for x in vol.lattice.lengths])
            cartesian_coords = vol.lattice.get_cartesian_coords(
                fractional_coords)
            self.cart_sites.append(tuple(cartesian_coords))

    def fractional_path(self, vol: Volume):
        """Convert voxel coordinates to fractional coordinates.

        Parameters
        ----------
        vol : Volume
            Volume object containing the grid information
        """
        if self.sites is None:
            raise ValueError('Voxel coordinates of the path are required.')
        # Manually get the fractional coordinates of the sites, then use
        # lattice.get_fractional_coords and make into tuple
        self.frac_sites = [
            tuple(vol.lattice.get_fractional_coords(site))
            for site in self.sites
        ]

    def wrap(self, F: np.ndarray):
        """Wrap path in periodic boundary conditions in-place.

        Parameters
        ----------
        F: np.ndarray
            Grid in which the path sites will be wrapped
        """
        if self.sites is None:
            raise ValueError('Voxel coordinates of the path are required.')

        X, Y, Z = F.shape
        self.sites = [(x % X, y % Y, z % Z) for x, y, z in self.sites]
        if self.cart_sites is not None:
            self.cart_sites = [(x % X, y % Y, z % Z)
                               for x, y, z in self.cart_sites]

    def path_over_structure(self, structure: Structure, vol: Volume):
        """Find the nearest site of the structure to the path sites.

        Parameters
        ----------
        structure : Structure
            Reference structure
        vol : Volume
            Volume object that contains the information about the nearest sites of the structure
        """

        if self.frac_sites is None:
            raise ValueError(
                'Fractional coordinates of the path are required.')
        if vol.nearest_structure_map is None or vol.nearest_structure_tree is None:
            raise ValueError(
                'Volume object must have nearest_structure_map and nearest_structure_tree attributes.'
            )

        # Get the indices of the nearest structure sites to the path sites
        nearest_structure_indices = [
            vol.nearest_structure_tree.query(site)[1]
            for site in self.frac_sites
        ]
        # and use it to get its label and coordinates
        self.nearest_structure_label = [
            structure.labels[vol.nearest_structure_map[index]]
            for index in nearest_structure_indices
        ]
        self.nearest_structure_coord = [
            structure.cart_coords[vol.nearest_structure_map[index]]
            for index in nearest_structure_indices
        ]

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
    def end_site(self) -> tuple[int, int, int]:
        """Return end site."""
        if self.sites is None:
            raise ValueError('Voxel coordinates of the path are required.')
        return self.sites[-1]


def free_energy_graph(F: np.ndarray,
                      max_energy_threshold: float = 1e20,
                      diagonal: bool = True) -> nx.Graph:
    """Compute the graph of the free energy for networkx functions.

    Parameters
    ----------
    F : np.ndarray
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
    for index, Fi in np.ndenumerate(F):
        if 0 <= Fi < max_energy_threshold:
            G.add_node(index, energy=Fi)
    for node in G.nodes:
        for move in movements:
            neighbor = tuple((node + move) % F.shape)
            if neighbor in G.nodes:
                exp_n_energy = np.exp(F[neighbor])
                if exp_n_energy < max_energy_threshold:
                    G.add_edge(node,
                               neighbor,
                               weight=F[neighbor],
                               weight_exp=exp_n_energy)
                else:
                    G.add_edge(node,
                               neighbor,
                               weight=F[neighbor],
                               weight_exp=max_energy_threshold)

    return G


_PATHFINDING_METHODS = Literal['dijkstra', 'bellman-ford', 'minmax-energy',
                               'dijkstra-exp']


def optimal_path(
    F_graph: nx.Graph,
    start: tuple,
    end: tuple,
    method: _PATHFINDING_METHODS = 'dijkstra',
) -> Pathway:
    """Calculate the shortest cost-effective path using the desired method.

    Parameters
    ----------
    F_graph : nx.Graph
        Graph of the free energy
    start : tuple
        Coordinates of the starting point
    end: tuple
        Coordinates of the ending point
    method : str
        Method used to calculate the shortest path. Options are:
        - 'dijkstra': Dijkstra's algorithm
        - 'bellman-ford': Bellman-Ford algorithm
        - 'minmax-energy': Minmax energy algorithm
        - 'dijkstra-exp': Dijkstra's algorithm with exponential weights

    Returns
    -------
    path: Pathway
        Optimal path on the graph between start and end
    """

    optimal_path = nx.shortest_path(
        F_graph,
        source=start,
        target=end,
        weight='weight_exp' if method == 'dijkstra-exp' else 'weight',
        method='dijkstra'
        if method == 'dijkstra-exp' or method == 'minmax-energy' else method)

    if method == 'dijkstra' or method == 'bellman-ford' or method == 'dijkstra-exp':

        path_energy = [F_graph.nodes[node]['energy'] for node in optimal_path]
        path = Pathway(sites=optimal_path, energy=path_energy)
        return path

    elif method == 'minmax-energy':
        max_energy = max(
            [F_graph.nodes[node]['energy'] for node in optimal_path])
        minmax_energy = max_energy
        pruned_F_graph = F_graph.copy()
        while minmax_energy <= max_energy:
            # Find the node of the path with the highest energy
            max_node = max(optimal_path,
                           key=lambda x: F_graph.nodes[x]['energy'])
            # remove this node from the graph
            pruned_F_graph.remove_node(max_node)
            # recompute the path
            pruned_path = nx.shortest_path(
                pruned_F_graph,
                source=start,
                target=end,
                weight='weight',
            )
            minmax_energy = max(
                [F_graph.nodes[node]['energy'] for node in pruned_path])

            if minmax_energy < max_energy:
                optimal_path = pruned_path
                max_energy = minmax_energy

        path_energy = [F_graph.nodes[node]['energy'] for node in optimal_path]
        path = Pathway(sites=optimal_path, energy=path_energy)
        return path

    else:
        raise ValueError(f'Unknown method {method}')


def find_best_perc_path(F: np.ndarray,
                        peaks: np.ndarray,
                        percolate_x: bool = True,
                        percolate_y: bool = False,
                        percolate_z: bool = False) -> Pathway:
    """Calculate the best percolating path.

    Parameters
    ----------
    F : np.ndarray
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
    xyz_real = F.shape

    # Find percolation using virtual images along the required dimensions
    if not any([percolate_x, percolate_y, percolate_z]):
        print('Warning: percolation is not defined')
        return Pathway()

    # Tile the grind in the percolation directions
    F = np.tile(F, (1 + percolate_x, 1 + percolate_y, 1 + percolate_z))

    # Get F on a graph
    F_graph = free_energy_graph(F, max_energy_threshold=1e7, diagonal=True)

    # reaching the percolating image
    image = tuple(
        x * px
        for x, px in zip(xyz_real, (percolate_x, percolate_y, percolate_z)))

    # Find the lowest cost path that percolates along the x dimension
    best_cost = float('inf')
    best_path = Pathway()

    for starting_point in peaks:

        # Get the end point which is a periodic image of the peak
        end_point = starting_point + image

        # Find the shortest percolating path through this peak
        try:
            path = optimal_path(
                F_graph,
                tuple(starting_point),
                tuple(end_point),
            )
        except nx.NetworkXNoPath:
            continue

        cost = path.cost

        if cost < best_cost:
            best_cost = cost
            best_path = path

    return best_path
