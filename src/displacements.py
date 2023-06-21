import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core import Element, Lattice
from pymatgen.core.trajectory import Trajectory


def calculate_cell_offsets(traj: Trajectory) -> np.ndarray:
    """Calculate cell offsets from trajectory."""
    assert not traj.coords_are_displacement
    coords = traj.coords
    return calculate_cell_offsets_from_coords(coords)


def calculate_cell_offsets_from_coords(coords: np.ndarray) -> np.ndarray:
    """Calculate cell offsets from starting position.

    For example, if a site is at [0, 0, 0.9] -> [0, 0, 0.1]
    assume it has jumped to the next cell: [0, 0, 1.1]

    Parameters
    ----------
    coords : np.ndarray[i, j, k]
        3-dimensional numpy array with dimensions i: time_steps, j: sites, k: coordinates

    Returns
    -------
    offsets : np.ndarray[i, j, k]
        Integer array with unit cell offset vectors.
    """
    first = coords[0, np.newaxis]
    diff = np.diff(coords, axis=0, prepend=first)

    digits = np.digitize(diff, bins=[0.5, -0.5]) - 1

    offsets = np.cumsum(digits, axis=0)
    return offsets


def calculate_lengths(vectors: np.ndarray,
                      metric_tensor: np.ndarray) -> np.ndarray:
    """Calculate vector lengths using the metric tensor (Dunitz 1078, p227).

    Parameters
    ----------
    vectors : np.ndarray[i, j, k]
        Vectors as in fractional coordinates
    metric_tensor : np.ndarray
        Metric tensor for the lattice

    Returns
    -------
    lengths : np.ndarray
        Vector lengths
    """
    tmp = np.dot(vectors, metric_tensor)
    total_displacement = np.einsum('ij,ji->i', tmp, vectors.T)
    assert total_displacement.shape[0] == vectors.shape[0]
    assert total_displacement.ndim == 1
    return np.sqrt(total_displacement)


def calculate_displacements(traj_coords: np.ndarray,
                            lattice: Lattice,
                            equilibration_steps: int = 0) -> np.ndarray:
    """Calculate displacements from first set of positions.

    Corrects for elements jumping to the next unit cell.

    Parameters
    ----------
    traj_coords : np.ndarray
        3-dimensional numpy array with dimensions i: time_steps, j: sites, k: coordinates
    lattice : Lattice
        Lattice parameters
    equilibration_steps : int, optional
        Number of steps to skip before equilibration

    Returns
    -------
    displacements : np.ndarray
        Displacements from first set of positions.
    """
    offsets = calculate_cell_offsets_from_coords(traj_coords)

    corrected_coords = traj_coords + offsets

    displacements = []

    first = corrected_coords[equilibration_steps]

    for disp in corrected_coords[equilibration_steps:]:
        diff_vectors = disp - first
        lengths = calculate_lengths(diff_vectors,
                                    metric_tensor=lattice.metric_tensor)
        displacements.append(lengths)

    displacements = np.array(displacements)

    return displacements


def plot_displacement_per_site(displacements: np.ndarray):
    """Plot displacement per site.

    Parameters
    ----------
    displacements : np.ndarray
        Numpy array with displacements
    """
    fig, ax = plt.subplots()

    for site_displacement in displacements.T:
        ax.plot(site_displacement, lw=0.3)

    ax.set(title='Displacement of diffusing element',
           xlabel='Time step',
           ylabel='Displacement (Angstrom)')

    plt.show()


def plot_displacement_per_element(species: list[Element],
                                  displacements: np.ndarray):
    """Plot displacement per element.

    Parameters
    ----------
    structure : Structure
        Pymatgen structure used for labelling
    displacements : np.ndarray
        Numpy array with displacements
    """
    from collections import defaultdict

    grouped = defaultdict(list)

    for specie, displacement in zip(species, displacements.T):
        grouped[specie.name].append(displacement)

    fig, ax = plt.subplots()

    for specie, displacement in grouped.items():
        mean_disp = np.mean(displacement, axis=0)
        ax.plot(mean_disp, lw=0.3, label=specie)

    ax.legend()
    ax.set(title='Displacement per element',
           xlabel='Time step',
           ylabel='Displacement (Angstrom)')

    plt.show()


def plot_displacement_histogram(displacements: np.ndarray):
    """Plot histogram of total displacement at final timestep.

    Parameters
    ----------
    displacements : np.ndarray
        Numpy array with displacements
    """
    fig, ax = plt.subplots()
    ax.hist(displacements[-1])
    ax.set(title='Histogram of displacement of diffusing element',
           xlabel='Displacement (Angstrom)',
           ylabel='Nr. of atoms')
    plt.show()


if __name__ == '__main__':
    from GEMDAT import load_project

    vasp_xml = '/run/media/stef/Scratch/md-analysis-matlab-example/vasprun.xml'

    # skip first timesteps
    equilibration_steps = 1250

    traj_coords, data = load_project(vasp_xml, diffusing_element='Li')

    species = data['species']
    lattice = data['lattice']

    print(species)
    print(lattice)
    print(traj_coords.shape)

    displacements = calculate_displacements(
        traj_coords, lattice=lattice, equilibration_steps=equilibration_steps)

    plot_displacement_per_site(displacements)

    plot_displacement_per_element(species, displacements)

    plot_displacement_histogram(displacements)
