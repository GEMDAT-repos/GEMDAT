from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from pymatgen.electronic_structure import plotter

if TYPE_CHECKING:
    from gemdat import SitesData
    from gemdat.trajectory import GemdatTrajectory


def plot_jumps_vs_distance(*,
                           trajectory: GemdatTrajectory,
                           sites: SitesData,
                           jump_res: float = 0.1,
                           **kwargs) -> plt.Figure:
    """Plot jumps vs. distance histogram.

    Parameters
    ----------
    trajectory : GemdatTrajectory
        Input trajectory
    sites : SitesData
        Input sites data
    jump_res : float, optional
        Resolution of the bins in Angstrom

    Returns
    -------
    plt.Figure
    """
    lattice = trajectory.get_lattice()
    structure = sites.structure
    pdist = lattice.get_all_distances(structure.frac_coords,
                                      structure.frac_coords)

    bin_max = (1 + pdist.max() // jump_res) * jump_res
    n_bins = int(bin_max / jump_res) + 1
    x = np.linspace(0, bin_max, n_bins)
    counts = np.zeros_like(x)

    bin_idx = np.digitize(pdist, bins=x)
    for idx, n in zip(bin_idx.flatten(), sites.transitions.flatten()):
        counts[idx] += n

    fig, ax = plt.subplots()

    ax.bar(x, counts, width=(jump_res * 0.8))

    ax.set(title='Frequency vs Occurence',
           xlabel='Frequency (Hz)',
           ylabel='Occurrence (a.u.)')

    return fig


def plot_jumps_vs_time(*,
                       trajectory: GemdatTrajectory,
                       sites: SitesData,
                       binsize: int = 500,
                       **kwargs) -> plt.Figure:
    """Plot jumps vs. distance histogram.

    Parameters
    ----------
    trajectory : GemdatTrajectory
        Input trajectory
    sites : SitesData
        Input sites data
    binsize : int, optional
        Width of each bin in number of time steps


    Returns
    -------
    plt.Figure
    """
    n_steps = len(trajectory)
    bins = np.arange(0, n_steps + binsize, binsize)

    fig, ax = plt.subplots()

    ax.hist(sites.all_transitions[:, 4], bins=bins, width=0.8 * binsize)

    ax.set(title='Jumps vs. time',
           xlabel='Time (steps)',
           ylabel='Number of jumps')

    return fig


def plot_collective_jumps(*, trajectory: GemdatTrajectory, sites: SitesData,
                          **kwargs) -> plt.Figure:
    """Plot collective jumps per jump-type combination.

    Parameters
    ----------
    trajectory : GemdatTrajectory
        Input trajectory
    sites : SitesData
        Input sites data

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots()

    mat = ax.imshow(sites.coll_matrix)

    ticks = range(len(sites.jump_names))

    ax.set_xticks(ticks, labels=sites.jump_names, rotation=90)
    ax.set_yticks(ticks, labels=sites.jump_names)

    fig.colorbar(mat, ax=ax)

    ax.set(title='Cooperative jumps per jump-type combination')

    return fig


def plot_jumps_3d(*, trajectory: GemdatTrajectory, sites: SitesData,
                  **kwargs) -> plt.Figure:
    """Plot jumps in 3D.

    Parameters
    ----------
    trajectory : GemdatTrajectory
        Input trajectory
    sites : SitesData
        Input sites data

    Returns
    -------
    plt.Figure
    """

    class LabelItems:

        def __init__(self, labels, coords):
            self.labels = labels
            self.coords = coords

        def items(self):
            yield from zip(self.labels, self.coords)

    coords = sites.structure.frac_coords
    lattice = trajectory.get_lattice()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    site_labels = LabelItems(sites.structure.labels, coords)

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
        count = sites.transitions[i, j] + sites.transitions[j, i]
        if count == 0:
            continue

        coord_i = coords[i]
        coord_j = coords[j]

        lw = 1 + np.log(count)

        _, image = lattice.get_distance_and_image(coord_i, coord_j)

        # NOTE: might need to plot `line = [coord_i - image, coord_j]` as well
        line = [coord_i, coord_j + image]

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
        xlabel="x' (ang)",
        ylabel="y' (ang)",
        zlabel="z' (ang)",
    )

    ax.set_aspect('equal')  # only auto is supported

    return fig


def plot_jumps_3d_animation(*,
                            trajectory: GemdatTrajectory,
                            sites: SitesData,
                            t_start: int,
                            t_stop: int,
                            decay: float = 0.05,
                            skip: int = 5,
                            interval: int = 20,
                            **kwargs):
    """Plot jumps in 3D as an animation over time.

    # TODO
    # - Refactor using init func
    # - Refactor shared code with `plot_jumps_3d`
    # - Save/export animation somehow
    # - Probably cleaner to combine these functions as a class

    Parameters
    ----------
    trajectory : GemdatTrajectory
        Input trajectory
    sites : SitesData
        Input sites data
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
    plt.Figure
    """
    minwidth = 0.2
    maxwidth = 5.0

    class LabelItems:

        def __init__(self, labels, coords):
            self.labels = labels
            self.coords = coords

        def items(self):
            yield from zip(self.labels, self.coords)

    coords = sites.structure.frac_coords
    lattice = trajectory.get_lattice()

    color_from = colormaps['Set1'].colors
    color_to = colormaps['Pastel1'].colors

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    LabelItems(sites.structure.labels, coords)

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

    time_col = 4
    event_order = sites.all_transitions[:, time_col].argsort()

    events = sites.all_transitions[event_order]

    for _, site_i, site_j, *_ in events:

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
        xlabel="x' (ang)",
        ylabel="y' (ang)",
        zlabel="z' (ang)",
    )

    ax.set_aspect('equal')  # only auto is supported

    def update(frame_no):
        t_frame = t_start + (frame_no * skip)

        for i, (atom, frm, to, _, t_jump) in enumerate(events):
            if t_jump > t_frame:
                break

            lw = max(maxwidth - decay * (t_frame - t_jump), minwidth)

            line = lines[i]
            line.set_color('red')
            line.set_linewidth(lw)

            points[frm].set_facecolor(color_from[atom % len(color_from)])
            points[to].set_facecolor(color_to[atom % len(color_to)])

        ax.set_title(f'T: {t_frame} | Next jump: {t_jump}')

    n_frames = int((t_stop - t_start) / skip)

    return animation.FuncAnimation(fig=fig,
                                   func=update,
                                   frames=n_frames,
                                   interval=interval,
                                   repeat=False)
