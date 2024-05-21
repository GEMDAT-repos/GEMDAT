from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from pymatgen.electronic_structure import plotter

if TYPE_CHECKING:
    from gemdat import Jumps


def jumps_vs_distance(
    *,
    jumps: Jumps,
    jump_res: float = 0.1,
) -> plt.Figure:
    """Plot jumps vs. distance histogram.

    Parameters
    ----------
    jumps : Jumps
        Input data
    jump_res : float, optional
        Resolution of the bins in Angstrom

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    sites = jumps.sites

    trajectory = jumps.trajectory
    lattice = trajectory.get_lattice()

    pdist = lattice.get_all_distances(sites.frac_coords, sites.frac_coords)

    bin_max = (1 + pdist.max() // jump_res) * jump_res
    n_bins = int(bin_max / jump_res) + 1
    x = np.linspace(0, bin_max, n_bins)
    counts = np.zeros_like(x)

    bin_idx = np.digitize(pdist, bins=x)
    for idx, n in zip(bin_idx.flatten(), jumps.matrix().flatten()):
        counts[idx] += n

    fig, ax = plt.subplots()

    ax.bar(x, counts, width=(jump_res * 0.8))

    ax.set(title='Jumps vs. Distance',
           xlabel='Distance (Å)',
           ylabel='Number of jumps')

    return fig


def jumps_vs_time(*, jumps: Jumps, binsize: int = 500) -> plt.Figure:
    """Plot jumps vs. time histogram.

    Parameters
    ----------
    jumps : Jumps
        Input data
    binsize : int, optional
        Width of each bin in number of time steps

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """

    trajectory = jumps.trajectory

    n_steps = len(trajectory)
    bins = np.arange(0, n_steps + binsize, binsize)

    fig, ax = plt.subplots()

    ax.hist(jumps.data['stop time'], bins=bins, width=0.8 * binsize)

    ax.set(title='Jumps vs. time',
           xlabel='Time (steps)',
           ylabel='Number of jumps')

    return fig


def collective_jumps(*, jumps: Jumps) -> plt.Figure:
    """Plot collective jumps per jump-type combination.

    Parameters
    ----------
    jumps : Jumps
        Input data

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    fig, ax = plt.subplots()
    matrix = jumps.collective().site_pair_count_matrix()
    labels = jumps.collective().site_pair_count_matrix_labels()

    mat = ax.imshow(matrix)

    ticks = range(len(labels))

    ax.set_xticks(ticks, labels=labels, rotation=90)
    ax.set_yticks(ticks, labels=labels)

    fig.colorbar(mat, ax=ax)

    ax.set(title='Cooperative jumps per jump-type combination')

    return fig


def jumps_3d(*, jumps: Jumps) -> plt.Figure:
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

    xyz_labels = LabelItems('OABC', [[-0.1, -0.1, -0.1], [1.1, -0.1, -0.1],
                                     [-0.1, 1.1, -0.1], [-0.1, -0.1, 1.1]])

    plotter.plot_lattice_vectors(lattice, ax=ax, linewidth=1)
    plotter.plot_labels(xyz_labels,
                        lattice=lattice,
                        ax=ax,
                        color='green',
                        size=12)
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
            plotter.plot_path(line,
                              lattice=lattice,
                              ax=ax,
                              color='red',
                              linewidth=lw)

    plotter.plot_labels(site_labels,
                        lattice=lattice,
                        ax=ax,
                        color='black',
                        size=8)

    ax.set(
        title='Jumps between sites',
        xlabel="x' (Å)",
        ylabel="y' (Å)",
        zlabel="z' (Å)",
    )

    ax.set_aspect('equal')  # only auto is supported

    return fig


def jumps_3d_animation(
    *,
    jumps: Jumps,
    t_start: int,
    t_stop: int,
    decay: float = 0.05,
    skip: int = 5,
    interval: int = 20,
) -> animation.FuncAnimation:
    """Plot jumps in 3D as an animation over time.

    Parameters
    ----------
    jumps : Jumps
        Input data
    t_start : int
        Time step to start animation (relative to equilibration time)
    t_stop : int
        Time step to stop animation (relative to equilibration time)
    decay : float, optional
        Controls the decay of the line width (higher = faster decay)
    skip : float, optional
        Skip frames (increase for faster, but less accurate rendering)
    interval : int, optional
        Delay between frames in milliseconds.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    minwidth = 0.2
    maxwidth = 5.0

    trajectory = jumps.trajectory

    class LabelItems:

        def __init__(self, labels, coords):
            self.labels = labels
            self.coords = coords

        def items(self):
            yield from zip(self.labels, self.coords)

    coords = jumps.sites.frac_coords
    lattice = trajectory.get_lattice()

    color_from = colormaps['Set1'].colors  # type: ignore
    color_to = colormaps['Pastel1'].colors  # type: ignore

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    xyz_labels = LabelItems('OABC', [[-0.1, -0.1, -0.1], [1.1, -0.1, -0.1],
                                     [-0.1, 1.1, -0.1], [-0.1, -0.1, 1.1]])

    plotter.plot_lattice_vectors(lattice, ax=ax, linewidth=1)

    plotter.plot_labels(xyz_labels,
                        lattice=lattice,
                        ax=ax,
                        color='green',
                        size=12)

    assert len(ax.collections) == 0
    plotter.plot_points(coords,
                        lattice=lattice,
                        ax=ax,
                        s=50,
                        color='white',
                        edgecolor='black')
    points = ax.collections

    events = jumps.data.sort_values('start time', ignore_index=True)

    for _, event in events.iterrows():
        site_i = event['start site']
        site_j = event['destination site']

        coord_i = coords[site_i]
        coord_j = coords[site_j]

        lw = 0

        _, image = lattice.get_distance_and_image(coord_i, coord_j)

        line = [coord_i, coord_j + image]

        plotter.plot_path(line,
                          lattice=lattice,
                          ax=ax,
                          color='red',
                          linewidth=lw)

    lines = ax.lines[3:]

    ax.set(
        title='Jumps between sites',
        xlabel="x' (Å)",
        ylabel="y' (Å)",
        zlabel="z' (Å)",
    )

    ax.set_aspect('equal')  # only auto is supported

    def update(frame_no):
        t_frame = t_start + (frame_no * skip)

        for i, event in events.iterrows():

            if event['start time'] > t_frame:
                break

            lw = max(maxwidth - decay * (t_frame - event['start time']),
                     minwidth)

            line = lines[i]
            line.set_color('red')
            line.set_linewidth(lw)

            points[event['start site']].set_facecolor(
                color_from[event['atom index'] % len(color_from)])
            points[event['destination site']].set_facecolor(
                color_to[event['atom index'] % len(color_to)])

        start_time = event['start time']
        ax.set_title(f'T: {t_frame} | Next jump: {start_time}')

    n_frames = int((t_stop - t_start) / skip)

    return animation.FuncAnimation(fig=fig,
                                   func=update,
                                   frames=n_frames,
                                   interval=interval,
                                   repeat=False)
