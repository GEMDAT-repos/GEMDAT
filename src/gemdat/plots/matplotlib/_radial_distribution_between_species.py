from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from typing import Collection

    import matplotlib.figure

    from gemdat import Trajectory


def radial_distribution_between_species(
    trajectory: Trajectory,
    specie_1: str | Collection[str],
    specie_2: str | Collection[str],
    max_dist: float = 5.0,
    resolution: float = 0.1,
) -> matplotlib.figure.Figure:
    """Calculate RDFs from specie_1 to specie_2.

    Parameters
    ----------
    trajectory: Trajectory
        Input trajectory.
    specie_1: str | list[str]
        Name of specie or list of species
    specie_2: str | list[str]
        Name of specie or list of species
    max_dist: float, optional
        Max distance for rdf calculation
    resolution: float, optional
        Width of the bins

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    coords_1 = trajectory.filter(specie_1).coords
    coords_2 = trajectory.filter(specie_2).coords
    lattice = trajectory.get_lattice()

    if coords_2.ndim == 2:
        num_time_steps = 1
        num_atoms, num_dimensions = coords_2.shape
    else:
        num_time_steps, num_atoms, num_dimensions = coords_2.shape

    particle_vol = num_atoms / lattice.volume

    all_dists = np.concatenate(
        [
            lattice.get_all_distances(coords_1[t, :, :], coords_2[t, :, :])
            for t in range(num_time_steps)
        ]
    )
    distances = all_dists.flatten()

    bins = np.arange(0, max_dist + resolution, resolution)
    rdf, _ = np.histogram(distances, bins=bins, density=False)

    def normalize(radius: np.ndarray) -> np.ndarray:
        """Normalize bin to volume."""
        shell = (radius + resolution) ** 3 - radius**3
        return particle_vol * (4 / 3) * np.pi * shell

    norm = normalize(bins)[:-1]
    rdf = rdf / norm

    fig, ax = plt.subplots()
    ax.plot(bins[:-1], rdf)
    ax.set(
        title=f'RDF between {specie_1} and {specie_2} per element',
        xlabel='Displacement (Å)',
        ylabel='Nr. of atoms',
    )
    return fig
