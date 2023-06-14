import numpy as np

stable_names = None


def calc_site_occups(sites):
    nr_stable = sites['succes'].shape[0]
    nr_diff_names = sites['names'].shape[0]
    nr_steps = sites['atoms'].shape[0]
    nr_atoms = sites['atoms'].shape[1]

    nr_stable_names = 0
    for i in range(nr_diff_names):
        test = sites['names'][i].split(':')
        if test[0] != 'Transition':
            nr_stable_names += 1
            stable_names[nr_stable_names - 1] = test

    site_count = np.zeros(nr_stable_names)
    sites_occup = np.zeros(nr_stable_names)
    occup_parts = np.zeros((nr_stable, sites['nr_parts']))
    atom_locations = np.zeros(nr_stable_names)

    for i in range(nr_stable):
        for j in range(nr_stable_names):
            if sites['site_names'][i] == stable_names[j]:
                sites_occup[j] += sites['occupancy'][i]
                occup_parts[j, :] += sites['occup_parts'][i, :]
                site_count[j] += 1

    atom_loc_parts = np.zeros((nr_stable_names, sites['nr_parts']))

    for j in range(nr_stable_names):
        atom_locations[j] = sites_occup[j] / (nr_steps * nr_atoms)
        for k in range(sites['nr_parts']):
            atom_loc_parts[j, k] = occup_parts[j, k] / (
                np.ceil(nr_steps / sites['nr_parts']) * nr_atoms)

    total_frac = np.sum(atom_locations)
    print('Fraction of time the diffusing atoms are at given sites:',
          total_frac)
    if total_frac < 0.75:
        print('WARNING! WARNING! WARNING!')
        print(
            'WARNING! Atoms spent more than 20% of the time at unknown locations!'
        )
        print(
            'Change or add site locations for a better description of the jump and diffusion process!'
        )
        print(
            'You can check if the sites are correct using the density plot (in plot_displacement.m)'
        )

    for j in range(nr_stable_names):
        sites_occup[j] /= (nr_steps * site_count[j])
        for k in range(sites['nr_parts']):
            occup_parts[j, k] /= (np.ceil(nr_steps / sites['nr_parts']) *
                                  site_count[j])

    return stable_names, sites_occup, atom_locations, occup_parts, atom_loc_parts
