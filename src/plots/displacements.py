import matplotlib.pyplot as plt
import numpy as np
from gemdat.trajectory import Trajectory


def displacement_per_site(*, trajectory: Trajectory, **kwargs):
    """Plot displacement per site.

    Parameters
    ----------
    trajectory : Trajectory
    trajectory class containing trajectories, displacements
    """
    fig, ax = plt.subplots()

    for site_displacement in trajectory.displacements():
        ax.plot(site_displacement, lw=0.3)

    ax.set(title='Displacement of diffusing element',
           xlabel='Time step',
           ylabel='Displacement (Angstrom)')

    return fig


def displacement_per_element(*, trajectory: Trajectory, **kwargs):
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

    trajectory.get_structure(0)
    species = trajectory.species
    displacements = trajectory.displacements()

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


def displacement_histogram(*, trajectory: Trajectory, **kwargs):
    """Plot histogram of total displacement at final timestep.

    Parameters
    ----------
    trajectory : Trajectory
        trajectories of elements for which to plot displacement
    """
    fig, ax = plt.subplots()
    ax.hist(trajectory.displacements()[:, -1])
    ax.set(title='Histogram of displacement of diffusing element',
           xlabel='Displacement (Angstrom)',
           ylabel='Nr. of atoms')

    return fig
