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
