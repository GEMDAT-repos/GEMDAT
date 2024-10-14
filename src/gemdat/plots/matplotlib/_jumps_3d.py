from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from pymatgen.electronic_structure import plotter

if TYPE_CHECKING:
    import matplotlib.figure

    from gemdat import Jumps


def jumps_3d(*, jumps: Jumps) -> matplotlib.figure.Figure:
    """Plot jumps in 3D.

    Parameters
    ----------
    jumps : Jumps
        Input data

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    trajectory = jumps.trajectory
    sites = jumps.sites

    class LabelItems:
        def __init__(self, labels, coords):
            self.labels = labels
            self.coords = coords

        def items(self):
            yield from zip(self.labels, self.coords)

    coords = sites.frac_coords
    lattice = trajectory.get_lattice()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    site_labels = LabelItems(jumps.sites.labels, coords)

    xyz_labels = LabelItems(
        'OABC',
        [
            [-0.1, -0.1, -0.1],
            [1.1, -0.1, -0.1],
            [-0.1, 1.1, -0.1],
            [-0.1, -0.1, 1.1],
        ],
    )

    plotter.plot_lattice_vectors(lattice, ax=ax, linewidth=1)
    plotter.plot_labels(xyz_labels, lattice=lattice, ax=ax, color='green', size=12)
    plotter.plot_points(coords, lattice=lattice, ax=ax)

    for i, j in zip(*np.triu_indices(len(coords), k=1)):
        count = jumps.matrix()[i, j] + jumps.matrix()[j, i]
        if count == 0:
            continue

        coord_i = coords[i]
        coord_j = coords[j]

        lw = 1 + np.log(count)

        length, image = lattice.get_distance_and_image(coord_i, coord_j)

        # NOTE: might need to plot `line = [coord_i - image, coord_j]` as well
        if np.any(image != 0):
            lines = [(coord_i, coord_j + image), (coord_i - image, coord_j)]
        else:
            lines = [(coord_i, coord_j)]

        for line in lines:
            plotter.plot_path(line, lattice=lattice, ax=ax, color='red', linewidth=lw)

    plotter.plot_labels(site_labels, lattice=lattice, ax=ax, color='black', size=8)

    ax.set(
        title='Jumps between sites',
        xlabel="x' (Å)",
        ylabel="y' (Å)",
        zlabel="z' (Å)",
    )

    ax.set_aspect('equal')  # only auto is supported

    return fig
