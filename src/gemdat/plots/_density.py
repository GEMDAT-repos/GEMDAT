import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from skimage import measure


def density(lattice,
            diff_positions,
            site_positions,
            site_labels,
            resolution=0.1,
            save_plot=True):
    """Returns 3D plot of probability distributing of diffusing atom.

    Parameters
    ----------
    lattice : pymatgen.core.lattice.Lattice
        Lattice parameters for the volume
    diff_positions : np.ndarray
        Input positions of diffusing atom as 3D numpy array
    site_positions : np.ndarray
        Input positions of sites as 2D numpy array
    site_labels : np.ndarray
        Input labels of sites as 1D numpy array
    resolution : float
        The resolution in Angstrom
    save_plot : bool
        The option to save the plot as .html object
    """

    num_dimensions, num_atoms, num_time_steps = diff_positions.shape

    # Find minimal and maximal x, y, z values in cart_pos
    min_coor = np.sum(lattice * (lattice < 0), axis=0)
    max_coor = np.sum(lattice * (lattice > 0), axis=0)
    length = np.ceil((max_coor - min_coor) / resolution).astype(int)

    # Create the lattice points
    origin = [-min_coor[0], -min_coor[1], -min_coor[2]]
    lat = lattice / resolution
    origin = np.array(origin) / resolution
    lat_orig = lat + origin

    # Initialize the figure
    fig = go.Figure()

    # Plot the lattice vectors
    for i in range(3):
        fig.add_trace(
            go.Scatter3d(x=[origin[0], lat_orig[i, 0]],
                         y=[origin[1], lat_orig[i, 1]],
                         z=[origin[2], lat_orig[i, 2]],
                         mode='lines',
                         line=dict(color='black')))

        for j in range(3):
            if i != j:
                fig.add_trace(
                    go.Scatter3d(
                        x=[lat_orig[i, 0], lat_orig[i, 0] + lat[j, 0]],
                        y=[lat_orig[i, 1], lat_orig[i, 1] + lat[j, 1]],
                        z=[lat_orig[i, 2], lat_orig[i, 2] + lat[j, 2]],
                        mode='lines',
                        line=dict(color='black')))

                for k in range(3):
                    if k != i and k != j:
                        fig.add_trace(
                            go.Scatter3d(x=[
                                lat_orig[i, 0] + lat[j, 0],
                                origin[0] + lat[i, 0] + lat[j, 0] + lat[k, 0]
                            ],
                                         y=[
                                             lat_orig[i, 1] + lat[j, 1],
                                             origin[1] + lat[i, 1] +
                                             lat[j, 1] + lat[k, 1]
                                         ],
                                         z=[
                                             lat_orig[i, 2] + lat[j, 2],
                                             origin[2] + lat[i, 2] +
                                             lat[j, 2] + lat[k, 2]
                                         ],
                                         mode='lines',
                                         line=dict(color='black')))

    # Create a density box and fill it
    box = np.zeros((length[0], length[1], length[2]))
    pos = np.zeros(3, dtype=int)

    for atom in range(num_atoms):
        for time in range(num_time_steps):
            for i in range(3):
                pos[i] = int(
                    (diff_positions[i, atom, time] - min_coor[i]) / resolution)
                if pos[i] < 0:
                    pos[i] = length[i] + pos[i]
            box[pos[0], pos[1], pos[2]] += 1

    # Smooth the density
    box_smooth = gaussian_filter(box, sigma=1.0)

    # Define the colors, isosurface values, and alpha
    colors = ['red', 'yellow', 'cyan']
    iso = [0.25, 0.10, 0.007]  # Adjust isosurface values as needed
    alpha = [0.7, 0.5, 0.35]  # Adjust transparency as needed

    # Plot isosurfaces
    for i in range(len(iso)):
        isoval = iso[i] * np.max(box_smooth)
        verts, faces, _, _ = measure.marching_cubes(box_smooth, level=isoval)
        fig.add_trace(
            go.Mesh3d(x=verts[:, 0],
                      y=verts[:, 1],
                      z=verts[:, 2],
                      i=faces[:, 0],
                      j=faces[:, 1],
                      k=faces[:, 2],
                      opacity=alpha[i],
                      color=colors[i],
                      showlegend=False))

    # Plot site positions
    for i in range(site_positions.shape[1]):
        point_size = 5
        color = px.colors.qualitative.G10[i]

        fig.add_trace(
            go.Scatter3d(x=[origin[0] + site_positions[0, i] / resolution],
                         y=[origin[1] + site_positions[1, i] / resolution],
                         z=[origin[2] + site_positions[2, i] / resolution],
                         mode='markers',
                         name=site_labels[i],
                         marker=dict(size=point_size,
                                     color=color,
                                     line=dict(width=2.5)),
                         showlegend=False))

    # Customize the plot layout
    fig.update_layout(title='Density of diffusing element',
                      scene=dict(aspectmode='manual',
                                 aspectratio=dict(x=2, y=1, z=1),
                                 xaxis_title='X (Angstrom)',
                                 yaxis_title='Y (Angstrom)',
                                 zaxis_title='Z (Angstrom)'),
                      legend=dict(orientation='h',
                                  yanchor='bottom',
                                  xanchor='left',
                                  x=0,
                                  y=-0.1),
                      showlegend=True,
                      margin=dict(l=0, r=0, b=0, t=0),
                      scene_camera=dict(up=dict(x=0, y=0, z=1),
                                        center=dict(x=0, y=0, z=0),
                                        eye=dict(x=-1, y=1, z=-0.6)))
    for trace in fig['data']:
        trace['showlegend'] = False
    if save_plot:
        fig.write_html('Density_plot.html')

    return fig
