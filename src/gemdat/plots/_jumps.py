from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import colormaps
from pymatgen.electronic_structure import plotter

if TYPE_CHECKING:

    from gemdat import SitesData


def jumps_vs_distance(
    *,
    sites: SitesData,
    jump_res: float = 0.1,
) -> plt.Figure:
    """Plot jumps vs. distance histogram.

    Parameters
    ----------
    sites : SitesData
        Input sites data
    jump_res : float, optional
        Resolution of the bins in Angstrom

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    structure = sites.structure

    trajectory = sites.trajectory
    lattice = trajectory.get_lattice()

    pdist = lattice.get_all_distances(structure.frac_coords,
                                      structure.frac_coords)

    bin_max = (1 + pdist.max() // jump_res) * jump_res
    n_bins = int(bin_max / jump_res) + 1
    x = np.linspace(0, bin_max, n_bins)
    counts = np.zeros_like(x)

    bin_idx = np.digitize(pdist, bins=x)
    for idx, n in zip(bin_idx.flatten(), sites.transitions.matrix().flatten()):
        counts[idx] += n

    fig, ax = plt.subplots()

    ax.bar(x, counts, width=(jump_res * 0.8))

    ax.set(title='Jumps vs. Distance',
           xlabel='Distance (Angstrom)',
           ylabel='Number of jumps')

    return fig


def jumps_vs_distance2(*,
                       sites: SitesData,
                       jump_res: float = 0.1,
                       n_parts: int = 1) -> go.Figure:
    """Plot jumps vs. distance histogram.

    Parameters
    ----------
    sites : SitesData
        Input sites data
    jump_res : float, optional
        Resolution of the bins in Angstrom
    n_parts : int
        Number of parts for error analysis

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Output figure
    """
    structure = sites.structure
    trajectory = sites.trajectory
    lattice = trajectory.get_lattice()

    pdist = lattice.get_all_distances(structure.frac_coords,
                                      structure.frac_coords)

    bin_max = (1 + pdist.max() // jump_res) * jump_res
    n_bins = int(bin_max / jump_res) + 1
    x = np.linspace(0, bin_max, n_bins)

    bin_idx = np.digitize(pdist, bins=x)
    data = []
    for transitions_part in sites.transitions.split(n_parts=n_parts,
                                                    n_steps=len(
                                                        sites.trajectory)):
        counts = np.zeros_like(x)
        for idx, n in zip(bin_idx.flatten(),
                          transitions_part.matrix().flatten()):
            counts[idx] += n
        for idx in range(n_bins):
            if counts[idx] > 0:
                data.append((x[idx], counts[idx]))

    df = pd.DataFrame(data=data, columns=['Displacement', 'count'])

    grouped = df.groupby(['Displacement'])
    mean = grouped.mean().reset_index().rename(columns={'count': 'mean'})
    std = grouped.std().reset_index().rename(columns={'count': 'std'})
    df = mean.merge(std, how='inner')

    df['specie'] = sites.floating_specie

    if n_parts == 1:
        fig = px.bar(df,
                     x='Displacement',
                     y='mean',
                     color='specie',
                     barmode='stack')
    else:
        fig = px.bar(df,
                     x='Displacement',
                     y='mean',
                     color='specie',
                     error_y='std',
                     barmode='stack')

    fig.update_layout(title='Jumps vs. Distance',
                      xaxis_title='Distance (Angstrom)',
                      yaxis_title='Number of jumps')

    return fig


def jumps_vs_time(*, sites: SitesData, binsize: int = 500) -> plt.Figure:
    """Plot jumps vs. distance histogram.

    Parameters
    ----------
    sites : SitesData
        Input sites data
    binsize : int, optional
        Width of each bin in number of time steps

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    trajectory = sites.trajectory

    n_steps = len(trajectory)
    bins = np.arange(0, n_steps + binsize, binsize)

    fig, ax = plt.subplots()

    ax.hist(sites.transitions.events[:, 4], bins=bins, width=0.8 * binsize)

    ax.set(title='Jumps vs. time',
           xlabel='Time (steps)',
           ylabel='Number of jumps')

    return fig


def jumps_vs_time2(*,
                   sites: SitesData,
                   bins: int = 8,
                   n_parts: int = 1) -> go.Figure:
    """Plot jumps vs. distance histogram.

    Parameters
    ----------
    sites : SitesData
        Input sites data
    bins : int, optional
        Number of bins
    n_parts : int
        Number of parts for error analysis

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """

    maxlen = len(sites.trajectory) / n_parts
    binsize = maxlen / bins + 1
    data = []

    for transitions_part in sites.transitions.split(n_parts=n_parts,
                                                    n_steps=len(
                                                        sites.trajectory)):
        data.append(
            np.histogram(transitions_part.events[:, 4],
                         bins=bins,
                         range=(0., maxlen))[0])

    df = pd.DataFrame(data=data)
    columns = [binsize / 2 + binsize * col for col in range(bins)]

    mean = [df[col].mean() for col in df.columns]
    std = [df[col].std() for col in df.columns]

    df = pd.DataFrame(data=zip(columns, mean, std),
                      columns=['time', 'count', 'std'])
    df['specie'] = sites.floating_specie

    if n_parts > 1:
        fig = px.bar(df, x='time', y='count', color='specie', error_y='std')
    else:
        fig = px.bar(df, x='time', y='count', color='specie')

    fig.update_layout(bargap=0.2)

    fig.update_layout(title='Jumps vs. time',
                      xaxis_title='Time (steps)',
                      yaxis_title='Number of jumps')

    return fig


def collective_jumps(*, sites: SitesData) -> plt.Figure:
    """Plot collective jumps per jump-type combination.

    Parameters
    ----------
    sites : SitesData
        Input sites data

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    fig, ax = plt.subplots()

    mat = ax.imshow(sites.collective().matrix())

    ticks = range(len(sites.jump_names))

    ax.set_xticks(ticks, labels=sites.jump_names, rotation=90)
    ax.set_yticks(ticks, labels=sites.jump_names)

    fig.colorbar(mat, ax=ax)

    ax.set(title='Cooperative jumps per jump-type combination')

    return fig


def jumps_3d(*, sites: SitesData) -> plt.Figure:
    """Plot jumps in 3D.

    Parameters
    ----------
    sites : SitesData
        Input sites data

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    trajectory = sites.trajectory

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

    site_labels = LabelItems(sites.site_labels, coords)

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
        count = sites.transitions.matrix()[i,
                                           j] + sites.transitions.matrix()[j,
                                                                           i]
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


def jumps_3d_animation(
    *,
    sites: SitesData,
    t_start: int,
    t_stop: int,
    decay: float = 0.05,
    skip: int = 5,
    interval: int = 20,
) -> animation.FuncAnimation:
    """Plot jumps in 3D as an animation over time.

    Parameters
    ----------
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
    fig : matplotlib.figure.Figure
        Output figure
    """
    minwidth = 0.2
    maxwidth = 5.0

    trajectory = sites.trajectory

    class LabelItems:

        def __init__(self, labels, coords):
            self.labels = labels
            self.coords = coords

        def items(self):
            yield from zip(self.labels, self.coords)

    coords = sites.structure.frac_coords
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

    time_col = 4
    event_order = sites.transitions.events[:, time_col].argsort()

    events = sites.transitions.events[event_order]

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
