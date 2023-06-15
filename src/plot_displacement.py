import matplotlib.pyplot as plt
import numpy as np
import pymatgen
import tqdm


def plot_element_displacement(structures: list[pymatgen.core.Structure]):
    """Plot displacement per element.

    Parameters
    ----------
    structures (list[pymatgen.core.Structure]):
            List of pymatgen structures
    """
    initial_s = structures[0]

    distances = np.zeros((len(structures), len(initial_s)))

    for i, s in enumerate(tqdm.tqdm(structures)):
        for j, e in enumerate(s):
            d = initial_s[j].distance(e)
            distances[i, j] = d

    fig, ax = plt.subplots()

    for i, _ in enumerate(initial_s):
        ax.plot(distances.T[i], lw=0.3)

    ax.set(title='Displacement per element',
           xlabel='Time step',
           ylabel='Displacement (Angstrom)')

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

    plot_element_displacement(vasp_data.structures[equilibration_steps:])
