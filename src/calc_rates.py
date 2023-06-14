import numpy as np


def calc_rates(sites, sim_data):
    # Calculate the jump rates between sites (and the standard deviation) and activation energies
    nr_stable = sites.succes.shape[0]
    nr_diff_names = sites.names.shape[0]
    nr_parts = sites.nr_parts

    # Find the stable names, transition states start with 'Transition'
    nr_stable_names = 0
    for i in range(nr_diff_names):
        test = sites.names[i].split(':')
        if not test[0] == 'Transition':
            nr_stable_names += 1

    nr_combi = nr_stable_names**2

    # The possible name-combinations
    counter = 0
    jump_names = []
    for site1 in range(nr_stable_names):
        for site2 in range(nr_stable_names):
            counter += 1
            jump_names.append(f'{sites.names[site1]}_to_{sites.names[site2]}')

    # Find the successful jumps and combine them by name
    jumps = np.zeros((nr_combi, nr_parts))
    for site1 in range(nr_stable):
        for site2 in range(nr_stable):
            for part in range(nr_parts):
                if sites.succes[site1, site2, part] > 0:
                    name = f'{sites.site_names[site1]}_to_{sites.site_names[site2]}'
                    name_nr = 0
                    for i in range(nr_combi):
                        if jump_names[i] == name:
                            name_nr = i
                    if name_nr != 0:
                        jumps[name_nr, part] += sites.succes[site1, site2,
                                                             part]
                    else:
                        print(
                            'ERROR! No name found for a jump! (in calc_rates.m)'
                        )

    # Calculate activation energies, jump rates per atom per second, and the standard deviation
    rates = np.zeros((nr_combi, 2))
    temp_e_act = np.zeros((nr_combi, nr_parts))
    e_act = np.zeros((nr_combi, 2))
    nr_diff_atoms = sites.atoms.shape[1]
    jump_freqs = jumps / (nr_diff_atoms * (sim_data.total_time / nr_parts))
    jumps_total = np.sum(jumps, axis=1)
    for i in range(nr_combi):
        rates[i, 0] = np.mean(jump_freqs[i, :])
        rates[i, 1] = np.std(jump_freqs[i, :])

    for i in range(nr_combi):
        names = jump_names[i].split('_to_')
        for j in range(nr_stable_names):
            if sites.stable_names[j] == names[0]:
                name_nr = j
        for part in range(nr_parts):
            if jumps[i, part] > 0:
                atom_percentage = sites.atom_loc_parts[name_nr, part]
                eff_rate = jumps[i, part] / (atom_percentage *
                                             sim_data.nr_diffusing *
                                             (sim_data.total_time / nr_parts))
                if names[0] == names[1]:
                    eff_rate /= 2
                temp_e_act[
                    i, part] = -np.log(eff_rate / sim_data.attempt_freq) * (
                        sim_data.k_boltzmann *
                        sim_data.temperature) / sim_data.e_charge
                if temp_e_act[i, part] < 0:
                    print('Warning! Negative activation energy found')
                    print(
                        'This is caused by a low value in sites.atom_loc_parts,'
                        ' check if this occupancy is realistic!')
            else:
                temp_e_act[i, part] = np.nan

        if np.sum(jumps[i, :]) > 0:
            atom_percentage = np.mean(sites.atom_loc_parts[name_nr, :])
            eff_rate = np.sum(jumps[i, :]) / (
                atom_percentage * sim_data.nr_diffusing * sim_data.total_time)
            if names[0] == names[1]:
                eff_rate /= 2
            e_act[i, 0] = -np.log(eff_rate / sim_data.attempt_freq) * (
                sim_data.k_boltzmann *
                sim_data.temperature) / sim_data.e_charge
            e_act[i, 1] = np.nanstd(temp_e_act[i, :])

    return jump_names, jumps_total, rates, e_act
