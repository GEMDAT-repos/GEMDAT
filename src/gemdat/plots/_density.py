from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from skimage import measure

if TYPE_CHECKING:
    from pymatgen.core import Lattice, Structure

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
    colors_by_site = {site: px.colors.qualitative.G10[i] for i, site in enumerate(np.unique(np.array(labels)))}
    colors = list(map(colors_by_site.get, labels))

    assert len(points) == len(labels)

    for i, (x, y, z) in enumerate(points):
        color = colors[i % len(colors)]
        label = labels[i]

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


def plot_structure(structure: Structure, *, fig: go.Figure):
    """Plot structure using plotly.

    Parameters
    ----------
    structure : Structure
        Input structure
    fig : go.Figure
        Plotly figure to add traces too
    """
    plot_points(structure.cart_coords, labels=structure.labels, fig=fig)
    plot_lattice_vectors(structure.lattice, fig=fig)


def plot_volume(
    volume: Volume,
    *,
    fig: go.Figure,
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
    colors : list[str], optional
        Adjust colors as needed. Length of color, isovals and alphavals must match
    isovals : list[float], optional
        Adjust isosurface values as needed
    alphavals : list[float], optional
        Adjust transparency as needed
    """
    data = volume.data
    data = gaussian_filter(data, sigma=1.0)

    assert len(colors) == len(isovals) == len(alphavals)

    for i, isoval in enumerate(isovals):
        isoval = isoval * np.max(data)
        verts, faces, _, _ = measure.marching_cubes(data, level=isoval)

        # Transform verts to cartesian system
        verts = (verts + 0.5) / np.array(data.shape)
        cart_verts = volume.lattice.get_cartesian_coords(verts)

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


def density(vol: Volume, structure = None) -> go.Figure:
    """Create density plot from volume and structure.

    Uses plotly as plotting backend.

    Arguments
    ---------
    vol : Volume
        Input volume
    structure : Structure, optional
        Input structure

    Returns
    -------
    fig : go.Figure
        Output as plotly figure
    """
    lattice = vol.lattice

    fig = go.Figure()

    plot_lattice_vectors(lattice, fig=fig)
    if structure:
        plot_points(lattice.get_cartesian_coords(structure.frac_coords), structure.labels, fig=fig)
    plot_volume(vol, fig=fig)

    fig.update_layout(title='Density',
                      scene={
                          'aspectmode': 'manual',
                          'aspectratio': {
                              'x': 2,
                              'y': 1,
                              'z': 1
                          },
                          'xaxis_title': 'X (Angstrom)',
                          'yaxis_title': 'Y (Angstrom)',
                          'zaxis_title': 'Z (Angstrom)'
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
                          'up': {
                              'x': 0,
                              'y': 0,
                              'z': 1
                          },
                          'center': {
                              'x': 0,
                              'y': 0,
                              'z': 0
                          },
                          'eye': {
                              'x': -1,
                              'y': 1,
                              'z': -0.6
                          }
                      })

    return fig
