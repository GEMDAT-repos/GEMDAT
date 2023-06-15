import matplotlib.pyplot as plt
import numpy as np
import pymatgen
import tqdm


def get_displacements(structures: list[pymatgen.core.Structure]) -> np.ndarray:
    """Calculate displacements from first time step.

    Parameters
    ----------
    structures : list[pymatgen.core.Structure]
        List of pymatgen structures

    Returns
    -------
    np.ndarray
        Description
    """
    initial_s = structures[0]

    displacements = np.zeros((len(structures), len(initial_s)))

    for i, s in enumerate(tqdm.tqdm(structures)):
        for j, e in enumerate(s):
            d = initial_s[j].distance(e)
            displacements[i, j] = d

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


def plot_displacement_per_element(structure: pymatgen.core.Structure,
                                  displacements: np.ndarray):
    """Plot displacement per element.

    Parameters
    ----------
    structure : pymatgen.core.Structure
        Pymatgen structure used for labelling
    displacements : np.ndarray
        Numpy array with displacements
    """
    from collections import defaultdict

    d = defaultdict(list)

    for site, displacement in zip(structure, displacements):
        d[site.species_string].append(displacement)

    fig, ax = plt.subplots()

    for specie, displacement in d.items():
        mean_disp = np.mean(displacement, axis=0)
        ax.plot(mean_disp, lw=0.3)

    ax.set(title='Displacement per element',
           xlabel='Time step',
           ylabel='Displacement (Angstrom)')

    plt.show()


def plot_displacement_histogram(displacements: np.ndarray):
    """Plot histogram of total displacement at final timestep.

    Parameters
    ----------
    structure : pymatgen.core.Structure
        Pymatgen structure used for labelling
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
    from pymatgen.io.vasp.outputs import Vasprun

    vasp_xml = 'S:/md-analysis-matlab-example/vasprun.xml'

    vasp_data = Vasprun(
        vasp_xml,
        parse_dos=False,
        parse_eigen=False,
        parse_projected_eigen=False,
        parse_potcar_file=False,
    )

    # skip first timesteps
    equilibration_steps = 1250

    structures = vasp_data.structures[equilibration_steps:]

    displacements = get_displacements(structures)

    plot_displacement_per_site(displacements)

    plot_displacement_per_element(structures[0], displacements)

    plot_displacement_histogram(displacements)
