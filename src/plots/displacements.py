import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core import Structure


def plot_displacement_per_site(*, diff_displacements: np.ndarray, **kwargs):
    """Plot displacement per site.

    Parameters
    ----------
    displacements : np.ndarray
        Numpy array with displacements
    """
    fig, ax = plt.subplots()

    for site_displacement in diff_displacements:
        ax.plot(site_displacement, lw=0.3)

    ax.set(title='Displacement of diffusing element',
           xlabel='Time step',
           ylabel='Displacement (Angstrom)')

    return fig


def plot_displacement_per_element(structure: Structure,
                                  displacements: np.ndarray, species,
                                  **kwargs):
    """Plot displacement per element.

    Parameters
    ----------
    structure : Structure
        Pymatgen structure used for labelling
    displacements : np.ndarray
        Numpy array with displacements
    species:
    """
    from collections import defaultdict

    grouped = defaultdict(list)

    for specie, displacement in zip(species, displacements):
        grouped[specie.name].append(displacement)

    fig, ax = plt.subplots()

    for specie, displacement in grouped.items():
        mean_disp = np.mean(displacement, axis=0)
        ax.plot(mean_disp, lw=0.3, label=specie)

    ax.legend()
    ax.set(title='Displacement per element',
           xlabel='Time step',
           ylabel='Displacement (Angstrom)')

    return fig


def plot_displacement_histogram(diff_displacements: np.ndarray, **kwargs):
    """Plot histogram of total displacement of diffusing element at final
    timestep.

    Parameters
    ----------
    diff_displacements : np.ndarray
        Numpy array with displacements of diffusing element
    """
    fig, ax = plt.subplots()
    ax.hist(diff_displacements[:, -1])
    ax.set(title='Histogram of displacement of diffusing element',
           xlabel='Displacement (Angstrom)',
           ylabel='Nr. of atoms')

    return fig


def plot_atom_density_3d(atom: str = 'Li', resolution: float = 0.2, **kwargs):
    lattice = data.structure.lattice

    frac_coords = data.trajectory_coords.reshape(-1, 3)

    cart_coords = lattice.get_cartesian_coords(frac_coords)

    x0, y0, z0 = cart_coords.min(axis=0)
    x1, y1, z1 = cart_coords.max(axis=0)

    nx = int(1 + (x1 - x0) // resolution)
    ny = int(1 + (y1 - y0) // resolution)
    nz = int(1 + (z1 - z0) // resolution)

    xbins = np.linspace(x0, x1, nx)
    ybins = np.linspace(y0, y1, ny)
    zbins = np.linspace(z0, z1, nz)

    print(x0, x1)
    print(y0, y1)
    print(z0, z1)
    print(nx, ny, nz)

    stack = np.vstack([
        np.digitize(cart_coords[:, 0], bins=xbins),
        np.digitize(cart_coords[:, 1], bins=ybins),
        np.digitize(cart_coords[:, 2], bins=zbins),
    ]).T

    volume = np.zeros((nx + 1, ny + 1, nz + 1), dtype=int)

    indices, counts = np.unique(stack, return_counts=True, axis=0)

    i, j, k = indices.T

    volume[i, j, k] = counts

    # Matplotlib is tricky here, k3d, mayavi, pyvista?

    import k3d

    plot = k3d.plot(grid=(x0, y0, z0, x1, y1, z1))

    plot += k3d.volume(volume.astype(np.float32), bounds=(x0, x1, y0, y1, z0, z1))

    plot.display()
