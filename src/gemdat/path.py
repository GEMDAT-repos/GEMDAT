"""This module contains classes for computing optimal and percolating paths
between sites."""

from __future__ import annotations

import networkx as nx
import numpy as np


class Pathway:
    """Container class for paths between sites.

    Attributes
    ----------
    path: list[tuple]
        List of coordinates of the path
    path_energy: list[float]
        Energy along the path
    """

    def __init__(self, sites: list[tuple], energy: list[float]):
        """Store event data for jumps and transitions between sites.

        Parameters
        ----------
        sites: list[tuple]
            List of coordinates of the sites definint the path
        energy: list[float]
            List of the energy along the path
        """
        self.sites = sites
        self.energy = energy

    def wrap(self, F: np.ndarray) -> Pathway:
        """Wrap path in periodic boundary conditions.

        Parameters
        ----------
        F: np.ndarray
            Grid in which the path sites will be wrapped
        """
        X, Y, Z = F.shape
        self.sites = [(x % X, y % Y, z % Z) for x, y, z in self.sites]

        return self

    @property
    def cost(self) -> float:
        """Calculate the path cost."""
        return np.sum(self.energy)


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
                G.add_edge(node, neighbor, weight=F[neighbor])

    return G


def optimal_path(
    F_graph: nx.Graph,
    start: tuple,
    end: tuple,
) -> Pathway:
    """Calculate the shortest cost-effective path from start to end using
    Networkx library.

    Parameters
    ----------
    F_graph : nx.Graph
        Graph of the free energy
    start : tuple
        Coordinates of the starting point
    end: tuple
        Coordinates of the ending point

    Returns
    -------
    path: Pathway
        Optimal path on the graph between start and end
    """

    optimal_path = nx.shortest_path(F_graph,
                                    source=start,
                                    target=end,
                                    weight='weight')
    path_energy = [F_graph.nodes[node]['energy'] for node in optimal_path]

    path = Pathway(optimal_path, path_energy)

    return path


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
        return Pathway([], [])

    # Tile the grind in the percolation directions
    F = np.tile(F, (1 + percolate_x, 1 + percolate_y, 1 + percolate_z))

    # Get F on a graph
    F_graph = free_energy_graph(F, max_energy_threshold=1e7, diagonal=True)

    # Find the lowest cost path that percolates along the x dimension
    best_cost = float('inf')

    # reaching the percolating image
    image = tuple(
        x * px
        for x, px in zip(xyz_real, (percolate_x, percolate_y, percolate_z)))

    best_path = Pathway([], [])

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
