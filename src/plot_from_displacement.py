import matplotlib.pyplot as plt
import numpy as np


def plots_from_displacement(sim_data, sites, resolution):
    # Plot displacement for each diffusing atom:
    fig, ax = plt.subplots()
    for i in range(int(sim_data['nr_atoms'])):
        if sim_data['diffusing_atom'][0, 0][i]:
            ax.plot(sim_data['displacement'][0, 0][i, :])
    ax.set(title='Displacement of diffusing element',
           xlabel='Time step',
           ylabel='Displacement (Angstrom)')

    # Plot histogram of final displacement of diffusing atoms:
    displacement = np.zeros(int(sim_data['nr_diffusing']))
    count = 0
    for i in range(int(sim_data['nr_atoms'])):
        if sim_data['diffusing_atom'][0, 0][i]:
            count += 1
            displacement[count - 1] = sim_data['displacement'][0, 0][i, -1]
    fig, ax = plt.subplots()
    ax.hist(displacement)
    ax.set(title='Histogram of displacement of diffusing element',
           xlabel='Displacement (Angstrom)',
           ylabel='Nr. of atoms')

    # Plot displacement per element:
    counter = 0
    disp_elem = np.zeros(
        (int(sim_data['nr_elements']), int(sim_data['nr_steps'])))
    fig, ax = plt.subplots()
    for i in range(int(sim_data['nr_elements'])):
        for j in range(
                counter,
                counter + int(sim_data['nr_per_element'][0, 0][i].item())):
            disp_elem[i, :] += sim_data['displacement'][0, 0][j, :]
        counter += int(sim_data['nr_per_element'][0, 0][i].item())
        disp_elem[i, :] /= sim_data['nr_per_element'][0, 0][i]
        ax.plot(disp_elem[i, :])
    ax.set(title='Displacement per element',
           xlabel='Time step',
           ylabel='Displacement (Angstrom)')
    plt.show()
