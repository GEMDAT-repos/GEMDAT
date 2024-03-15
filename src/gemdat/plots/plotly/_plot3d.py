from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from skimage import measure

if TYPE_CHECKING:
    from pymatgen.core import Lattice, Structure

    from gemdat.jumps import Jumps
    from gemdat.path import Pathway
    from gemdat.volume import Volume


def plot_lattice_vectors(lattice: Lattice, *, fig: go.Figure):
    """Plot lattice using plotly.

    Parameters
    ----------
    lattice : Lattice
        Input lattice
    fig : go.Figure
        Plotly figure to add traces to
    """
    org = lattice.get_cartesian_coords([0.0, 0.0, 0.0])
    a = lattice.get_cartesian_coords([1.0, 0.0, 0.0])
    b = lattice.get_cartesian_coords([0.0, 1.0, 0.0])
    c = lattice.get_cartesian_coords([0.0, 0.0, 1.0])
    abc = a + b + c

    for i, (start, stop) in enumerate(
        ((org, a), (org, b), (org, c), (a, a + b), (a, a + c), (b, b + a),
         (b, b + c), (c, c + a), (c, c + b), (abc, abc - a), (abc, abc - b),
         (abc, abc - c))):

        color = ('red', 'blue', 'green', 'black')[min(i, 3)]
        width = (5, 5, 5, 1)[min(i, 3)]

        fig.add_trace(
            go.Scatter3d(
                x=(start[0], stop[0]),
                y=(start[1], stop[1]),
                z=(start[2], stop[2]),
                mode='lines',
                name=None,
                line={
                    'color': color,
                    'width': width
                },
                showlegend=False,
            ))


def plot_points(points: np.ndarray,
                labels: Sequence,
                *,
                fig: go.Figure,
                point_size: int = 5):
    """Plot points using plotly.

    Parameters
    ----------
    points : np.ndarray
        Input points
    labels : Sequence
        Labels for points. Length must match points.
    fig : go.Figure
        Plotly figure to add traces to
    point_size : int, optional
        Size of the points
    """
    assert len(points) == len(labels)

    colors = {
        label: px.colors.sample_colorscale('rainbow', [i / (len(labels) - 1)])
        for i, label in enumerate(labels)
    }

    for i, (x, y, z) in enumerate(points):
        label = labels[i]
        color = colors[label]

        fig.add_trace(
            go.Scatter3d(x=[x],
                         y=[y],
                         z=[z],
                         mode='markers',
                         name=label,
                         marker={
                             'size': point_size,
                             'color': color,
                             'line': {
                                 'width': 2.5
                             }
                         },
                         showlegend=False))


def plot_structure(structure: Structure,
                   *,
                   lattice: Lattice | None = None,
                   fig: go.Figure):
    """Plot structure using plotly.

    Parameters
    ----------
    structure : Structure
        Input structure
    lattice : Lattice | None
        If specified, use this lattic instead of `structure.lattice`.
    fig : go.Figure
        Plotly figure to add traces too
    """
    if lattice:
        cart_coords = lattice.get_cartesian_coords(structure.frac_coords)
    else:
        cart_coords = structure.cart_coords

    plot_points(cart_coords, labels=structure.labels, fig=fig)
    plot_lattice_vectors(structure.lattice, fig=fig)


def plot_volume(
    volume: Volume,
    *,
    fig: go.Figure,
    lattice: Lattice | None = None,
    colors: list[str] = ['red', 'yellow', 'cyan'],
    isovals: list[float] = [0.25, 0.10, 0.007],
    alphavals: list[float] = [0.6, 0.3, 0.15],
):
    """Plot volume using plotly.

    Parameters
    ----------
    volume : Volume
        Input volume
    fig : go.Figure
        Plotly figure to add traces too
    lattice : Lattice | None
        Use this lattice instead of `Volume.lattice`
    colors : list[str], optional
        Adjust colors as needed. Length of color, isovals and alphavals must match
    isovals : list[float], optional
        Adjust isosurface values as needed
    alphavals : list[float], optional
        Adjust transparency as needed
    """
    if lattice is None:
        lattice = volume.lattice

    data = volume.data
    data = gaussian_filter(data, sigma=1.0)

    assert len(colors) == len(isovals) == len(alphavals)

    for i, isoval in enumerate(isovals):
        isoval = isoval * np.max(data)
        verts, faces, _, _ = measure.marching_cubes(data, level=isoval)

        # Transform verts to cartesian system
        verts = (verts + 0.5) / np.array(data.shape)
        cart_verts = lattice.get_cartesian_coords(verts)

        fig.add_trace(
            go.Mesh3d(x=cart_verts[:, 0],
                      y=cart_verts[:, 1],
                      z=cart_verts[:, 2],
                      i=faces[:, 0],
                      j=faces[:, 1],
                      k=faces[:, 2],
                      name=f'{isoval=}',
                      opacity=alphavals[i],
                      color=colors[i],
                      showlegend=False))


def plot_paths(
    paths: Pathway | list[Pathway],
    *,
    volume: Volume,
    fig: go.Figure,
):
    """Ploth paths over free energy.

    Arguments
    ---------
    paths : Pathway | list[Pathway]
        Pathway object containing the energy along the path, or list of Pathways
    volume : Volume
        Input volume to create the landscape
    fig : go.Figure
        Plotly figure to add paths too
    """
    if isinstance(paths, list):
        optimal_path = paths[0]
    else:
        optimal_path = paths

    x_path, y_path, z_path = np.asarray(optimal_path.cartesian_path(volume)).T

    fig.add_trace(
        go.Scatter3d(
            x=x_path,
            y=y_path,
            z=z_path,
            mode='markers+lines',
            line={'width': 3},
            marker={
                'size': 6,
                'color': 'teal',
                'symbol': 'circle',
                'opacity': 0.9
            },
            name='Optimal path',
        ))

    # If available, plot the other pathways
    if isinstance(paths, list):
        for idx, path in enumerate(paths[1:]):
            x_path, y_path, z_path = np.asarray(path.cartesian_path(volume)).T

            fig.add_trace(
                go.Scatter3d(
                    x=x_path,
                    y=y_path,
                    z=z_path,
                    mode='markers+lines',
                    line={'width': 3},
                    marker={
                        'size': 5,
                        #'color': color,
                        'symbol': 'circle',
                        'opacity': 0.9
                    },
                    name=f'Alternative {idx+1}',
                ))


def plot_jumps(jumps: Jumps, *, fig: go.Figure):
    """Ploth jumps in 3D.

    Arguments
    ---------
    paths : Jumps
        Jumps object containing the jumps to plot
    fig : go.Figure
        Plotly figure to add traces too
    """
    coords = jumps.sites.frac_coords
    lattice = jumps.trajectory.get_lattice()

    for i, j in zip(*np.triu_indices(len(coords), k=1)):
        count = jumps.matrix()[i, j] + jumps.matrix()[j, i]
        if count == 0:
            continue

        coord_i = tuple(coords[i].tolist())
        coord_j = tuple(coords[j].tolist())

        lw = 1 + np.log(count)

        length, image = lattice.get_distance_and_image(coord_i, coord_j)

        if np.any(image != 0):
            lines = [(coord_i, coord_j + image), (coord_i - image, coord_j)]
        else:
            lines = [(coord_i, coord_j)]

        for line in lines:
            line = lattice.get_cartesian_coords(line)
            line_t = [_ for _ in zip(*line)]  # transpose, but pythonic

            fig.add_trace(
                go.Scatter3d(
                    x=line_t[0],
                    y=line_t[1],
                    z=line_t[2],
                    mode='lines',
                    showlegend=False,
                    line_dash='dashdot' if any(image) != 0 else 'solid',
                    line_width=lw * 3,
                    line_color='black',
                ))


def update_layout(*,
                  lattice: Lattice,
                  fig: go.Figure,
                  title: str = 'Gemdat 3D plot',
                  zoom: float = 0.1):
    """Update layout, title, scene, etc for figure.

    Arguments
    ---------
    lattice : Lattice
        Lattice information to determine the aspect ratio and camera position
    fig : go.Figure
        Plotly figure to update layout of
    zoom : float, optional
        Zoom level
    title : str
        Title of the plot
    """
    fig.update_layout(title=title,
                      scene={
                          'aspectmode': 'manual',
                          'aspectratio': {
                              'x': lattice.a * zoom,
                              'y': lattice.b * zoom,
                              'z': lattice.c * zoom,
                          },
                          'xaxis_title': 'X (Ångstrom)',
                          'yaxis_title': 'Y (Ångstrom)',
                          'zaxis_title': 'Z (Ångstrom)'
                      },
                      legend={
                          'orientation': 'h',
                          'yanchor': 'bottom',
                          'xanchor': 'left',
                          'x': 0,
                          'y': -0.1
                      },
                      showlegend=True,
                      margin={
                          'l': 0,
                          'r': 0,
                          'b': 0,
                          't': 0
                      },
                      scene_camera={
                          'projection': {
                              'type': 'orthographic'
                          },
                          'eye': {
                              'x': -lattice.a * 0.05,
                              'y': -lattice.b * 0.2,
                              'z': lattice.c * 0.15,
                          }
                      })


def plot_3d(*,
            volume: Volume | None = None,
            structure: Structure | None = None,
            paths: Pathway | list[Pathway] | None = None,
            jumps: Jumps | None = None,
            lattice: Lattice | None = None,
            title: str = '3D plot') -> go.Figure:
    """Plot 3d."""
    fig = go.Figure()

    if not lattice:
        if volume:
            lattice = volume.lattice
        elif structure:
            lattice = structure.lattice
        elif jumps:
            lattice = jumps.trajectory.get_lattice()
    else:
        raise ValueError('Cannot derive lattice from input.')

    plot_lattice_vectors(lattice, fig=fig)

    if volume:
        plot_volume(volume, lattice=lattice, fig=fig)

    if structure:
        plot_structure(structure=structure, lattice=lattice, fig=fig)

    if paths:
        # TODO: Does this need the volume?
        # Revise after https://github.com/GEMDAT-repos/GEMDAT/pull/282
        if not volume:
            raise NotImplementedError
        plot_paths(paths=paths, volume=volume, fig=fig)

    if jumps:
        plot_jumps(jumps=jumps, fig=fig)

    update_layout(title=title, lattice=lattice, fig=fig)

    return fig
