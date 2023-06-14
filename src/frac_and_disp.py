import numpy as np


def frac_and_disp(sim_data):
    print(
        'Transforming cartesian to fractional coordinates and calculating displacement'
    )
    sim_data['frac_pos'] = np.zeros(
        (3, sim_data['nr_atoms'], sim_data['nr_steps']))
    sim_data['displacement'] = np.zeros(
        (sim_data['nr_atoms'], sim_data['nr_steps']))

    d = np.zeros(3)
    for atom in range(sim_data['nr_atoms']):
        uc = np.zeros(
            3)  # Keep track of the amount of unit cells the atom has moved
        # For the first time step:
        sim_data['frac_pos'][:, atom,
                             0] = sim_data['cart_pos'][:, atom,
                                                       0] / sim_data['lattice']
        start = sim_data['cart_pos'][:, atom, 0]  # Starting position
        for time in range(1, sim_data['nr_steps']):
            sim_data['frac_pos'][:, atom, time] = sim_data[
                'cart_pos'][:, atom, time] / sim_data['lattice']
            for i in range(3):
                frac_diff = sim_data['frac_pos'][
                    i, atom, time] - sim_data['frac_pos'][i, atom, time - 1]
                # If frac_pos differs by more than 0.5 from the previous time step, the atom changed unit cell
                if frac_diff > 0.5:
                    uc[i] = uc[i] - 1
                elif frac_diff < -0.5:
                    uc[i] = uc[i] + 1
            # Calculate displacement:
            for i in range(3):
                d[i] = (sim_data['cart_pos'][i, atom, time] - start[i] +
                        uc.dot(sim_data['lattice'][:, i]))**2
            sim_data['displacement'][atom, time] = np.sqrt(np.sum(d))

    return sim_data
