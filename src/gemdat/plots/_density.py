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
    o = lattice.get_cartesian_coords([0.0, 0.0, 0.0])
    a = lattice.get_cartesian_coords([1.0, 0.0, 0.0])
    b = lattice.get_cartesian_coords([0.0, 1.0, 0.0])
    c = lattice.get_cartesian_coords([0.0, 0.0, 1.0])
    abc = a + b + c

    for start, stop in ((o, a), (o, b), (o, c), (a, a + b), (a, a + c),
                        (b, b + a), (b, b + c), (c, c + a), (c, c + b),
                        (abc, abc - a), (abc, abc - b), (abc, abc - c)):
        fig.add_trace(
            go.Scatter3d(
                x=(start[0], stop[0]),
                y=(start[1], stop[1]),
                z=(start[2], stop[2]),
                mode='lines',
                line={'color': 'black'},
                showlegend=False,
            ))

    return fig


def plot_points(points: np.ndarray, labels: Sequence, *, fig: go.Figure):
    point_size = 5
    colors = px.colors.qualitative.G10

    for i, (x, y, z) in enumerate(points):
        color = colors[i % len(colors)]

        fig.add_trace(
            go.Scatter3d(x=[x],
                         y=[y],
                         z=[z],
                         mode='markers',
                         name=labels[i],
                         marker={
                             'size': point_size,
                             'color': color,
                             'line': {
                                 'width': 2.5
                             }
                         },
                         showlegend=False))
    return fig


def plot_structure(structure: Structure, *, fig: go.Figure):
    plot_points(structure.cart_coords, labels=structure.labels, fig=fig)
    plot_lattice_vectors(structure.lattice, fig=fig)


def plot_volume(volume, *, fig):
    data = volume.data
    # data = np.pad(data, ((0, 1), (0,1), (0,1)), mode='wrap')
    data = gaussian_filter(data, sigma=1.0)

    colors = ['red', 'yellow', 'cyan']
    iso_values = [0.25, 0.10, 0.007]  # Adjust isosurface values as needed
    alpha_values = [0.7, 0.5, 0.35]  # Adjust transparency as needed

    # Plot isosurfaces
    for i, isoval in enumerate(iso_values):
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
                      opacity=alpha_values[i],
                      color=colors[i],
                      showlegend=False))

    return fig


def density(vol: Volume, structure: Structure) -> go.Figure:
    lattice = structure.lattice

    fig = go.Figure()
    plot_lattice_vectors(lattice, fig=fig)
    plot_points(structure.cart_coords, structure.labels, fig=fig)
    plot_volume(vol, fig=fig)

    fig.update_layout(title='Density of diffusing element',
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
