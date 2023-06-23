import matplotlib.pyplot as plt
import numpy as np

from ..data import Data


def plot_displacement_per_site(data: Data):
    """Plot displacement per site.

    Parameters
    ----------
    displacements : np.ndarray
        Numpy array with displacements
    """
    fig, ax = plt.subplots()

    for site_displacement in data.displacements:
        ax.plot(site_displacement, lw=0.3)

    ax.set(title='Displacement of diffusing element',
           xlabel='Time step',
           ylabel='Displacement (Angstrom)')

    plt.show()


def plot_displacement_per_element(data: Data):
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

    for specie, displacement in zip(data.species, data.displacements):
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


def plot_displacement_histogram(data: Data):
    """Plot histogram of total displacement at final timestep.

    Parameters
    ----------
    displacements : np.ndarray
        Numpy array with displacements
    """
    fig, ax = plt.subplots()
    ax.hist(data.displacements[:, -1])
    ax.set(title='Histogram of displacement of diffusing element',
           xlabel='Displacement (Angstrom)',
           ylabel='Nr. of atoms')
    plt.show()
