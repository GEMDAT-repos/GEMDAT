import numpy as np

from .calc_dist_sqrd_frac import calc_dist_sqrd_frac
from .frac_to_cart import frac_to_cart

known_materials = None


def find_sites(sim_data, material, nr_parts):
    print('Finding site occupations for given structure')
    finished = False  # Check if the function has finished correctly

    # Get the names and positions for the given material
    names, sites_pos, supercell = known_materials(material)
    sites = {}
    sites['names'] = np.unique(names)
    sites['supercell'] = supercell

    # Define some things
    dist_close = 2 * sim_data['vibration_amp']  # 'amplitude of vibration'
    nr_sites = sites_pos.shape[1]
    part_size = np.ceil(sim_data['nr_steps'] / nr_parts)
    lattice = sim_data['lattice']

    # The cartesian positions of the given sites
    cart_pos = np.zeros((3, nr_sites))
    for i in range(nr_sites):
        cart_pos[:, i] = frac_to_cart(sites_pos[:, i], lattice)

    # Names of the sites
    nr_per_name = np.zeros(len(sites['names']))
    for i in range(len(names)):
        for j in range(len(sites['names'])):
            if sites['names'][j] == names[i]:
                nr_per_name[j] += 1
    sites['nr_per_name'] = nr_per_name

    # Add the names of transition sites
    unique_names = np.unique(names)
    # Single names
    for i in range(len(unique_names)):
        trans_name = 'Transition:' + unique_names[i]
        len(sites['names'])
        sites['names'] = np.append(sites['names'], trans_name)
    # All combinations
    for i in range(len(unique_names)):
        for j in range(len(unique_names)):
            trans_name = 'Transition:' + unique_names[i] + '-' + unique_names[j]
            len(sites['names'])
            sites['names'] = np.append(sites['names'], trans_name)

    # Loop over all possible positions, to see how close they are to each other
    # Should be less as the 2*dist_close to prevent overlap, otherwise it will be adjusted
    for site1 in range(nr_sites - 1):
        pos1 = sites_pos[:, site1]
        for site2 in range(site1 + 1, nr_sites):
            pos2 = sites_pos[:, site2]
            dist = np.sqrt(calc_dist_sqrd_frac(pos1, pos2, lattice))
            if dist < 2 * dist_close:
                print(
                    'WARNING! Crystallographic sites are overlapping with the chosen dist_close'
                    f' ({dist_close}), making dist_close smaller!')
                dist_close = 0.5 * (dist - 0.01)
                print(f'dist_close changed to {dist_close}')
                if dist_close < 0.25:
                    print(
                        'ERROR! ERROR! Two crystallographic sites are within half an Angstrom of each other'
                    )
                    print(
                        'ERROR! This is NOT realistic, check/change the given crystallographic site'
                        ' locations in known_materials.m!')
                    print(
                        f'ERROR! Sites number {site1} and {site2} are overlapping'
                    )
                    print(
                        f'ERROR! Coordinates are: ({pos1[0]}, {pos1[1]}, {pos1[2]}) and'
                        f' ({pos2[0]}, {pos2[1]}, {pos2[2]})')
                    return sites, finished

    close_sqrd = dist_close * dist_close  # to avoid taking the root everytime
    # Necessary matrices
    atom_site = np.zeros((sim_data['nr_steps'], sim_data['nr_diffusing']
                          ))  # A value of zero means it's not at a known site
    site_occup = np.zeros((nr_sites, nr_parts))
    transitions = np.zeros((nr_sites, nr_sites))
    trans_counter = 0
    all_trans = np.zeros((1, 4))
    succes = np.zeros((nr_sites, nr_sites, nr_parts))

    # Loop over all times and diffusing atoms
    diff_atom = 0
    part = 1
    end_step = sim_data['nr_steps']
    print('Determining the site positions of all diffusing atoms:')
    for atom in range(sim_data['nr_atoms']):
        if sim_data['diffusing_atom'][atom]:  # Only for the diffusing atoms
            diff_atom += 1
            # First find the first site this atom at
            not_found = True
            site = 0
            time_start = 1
            while not_found and time_start < end_step:
                if site < nr_sites:
                    site += 1
                else:
                    time_start += 1
                    site = 1
                pos_atom = sim_data['frac_pos'][:, atom, time_start]
                dist = calc_dist_sqrd_frac(pos_atom, sites_pos[:, site],
                                           lattice)
                if dist < close_sqrd:
                    not_found = False
                    # Atom is at this site
                    atom_site[time_start, diff_atom] = site
                    # Increase occupancy of this site
                    site_occup[site, part] += 1
                    prev_site = site  # Update previous site for next transition

            # After the first site has been found
            for time in range(time_start + 1, sim_data['nr_steps']):
                # Divide the simulation in multiple parts for statistics
                part = np.ceil(time / part_size)
                pos_atom = sim_data['frac_pos'][:, atom, time]
                # First check the site found at the last time_step, since it's the most likely position
                dist = calc_dist_sqrd_frac(pos_atom, sites_pos[:, prev_site],
                                           lattice)
                if dist < close_sqrd:
                    atom_site[time,
                              diff_atom] = prev_site  # Atom is at this site
                    site_occup[prev_site,
                               part] += 1  # Increase occupancy of this site
                    trans_start = time + 1
                else:  # Not found at the previous position
                    # Loop over all sites, exit the loop once a site has been found
                    not_found = True
                    site = 0
                    while not_found and site < nr_sites:
                        site += 1
                        dist = calc_dist_sqrd_frac(pos_atom,
                                                   sites_pos[:, site], lattice)
                        if dist < close_sqrd:
                            not_found = False
                            # Atom is at this site
                            atom_site[time, diff_atom] = site
                            # Increase occupancy of this site
                            site_occup[site, part] += 1

                    # If a transition happened, remember some things
                    if not not_found:  # Transition happened, i.e. the atom is found at a new site
                        transitions[prev_site, site] += 1
                        trans_counter += 1
                        all_trans[trans_counter, 0] = diff_atom  # the atom
                        all_trans[trans_counter,
                                  1] = prev_site  # the starting position
                        all_trans[trans_counter,
                                  2] = site  # the final position
                        all_trans[
                            trans_counter,
                            3] = trans_start  # the time the atom left its previous position
                        all_trans[
                            trans_counter,
                            4] = time  # the timestep (at the end of the transition)
                        succes[prev_site, site, part] += 1
                        prev_site = site

            print('*', end='')  # One atom has been finished...

    print('\n')  # Go to the next line in output
    print('Number of transitions between sites:', trans_counter)

    # Also assign the (failed) transition states with a site-number and name
    diff_atom = 0
    time_start = 0
    nr_trans_found = 0

    for atom in range(sim_data['nr_atoms']):
        if sim_data['diffusing_atom'][atom]:  # Only for the diffusing atoms
            diff_atom += 1
            prev_site = 0
            # What to do with atoms starting outside a given site?
            for time in range(time_start + 1, sim_data['nr_steps']):
                if atom_site[time, diff_atom] == 0 and prev_site == 0:
                    # It left the site
                    if time != 1:
                        prev_site = atom_site[time - 1, diff_atom]
                        trans_start = time
                elif atom_site[time, diff_atom] != 0 and prev_site != 0:
                    # It appeared at a new site
                    now_site = atom_site[time, diff_atom]
                    trans_end = time - 1
                    # Determine the 'trans_site_nr'
                    if now_site == prev_site:
                        trans_site_nr = nr_sites + now_site
                        names[trans_site_nr] = 'Transition:' + names[now_site]
                        atom_site[trans_start:trans_end,
                                  diff_atom] = trans_site_nr
                        prev_site = 0
                    else:  # loop over all transitions already found
                        if nr_trans_found == 0:  # For the first transition site
                            trans_found = np.zeros((1, 2))
                            trans_found[0, 0] = prev_site
                            trans_found[0, 1] = now_site
                            nr_trans_found = 1
                            trans_nr = 1
                        else:
                            trans_nr = 0
                            # Check if it has been found earlier
                            for i in range(nr_trans_found):
                                if trans_found[i,
                                               0] == prev_site and trans_found[
                                                   i, 1] == now_site:
                                    trans_nr = i + 1
                            if trans_nr == 0:  # Not found in previous step, so it is a NEW one
                                nr_trans_found += 1
                                trans_found = np.vstack(
                                    (trans_found, [prev_site, now_site]))
                                trans_nr = nr_trans_found
                        # Assign trans_site_nr etc.
                        trans_site_nr = 2 * nr_sites + trans_nr
                        atom_site[trans_start:trans_end,
                                  diff_atom] = trans_site_nr
                        names[trans_site_nr] = 'Transition:' + names[
                            prev_site] + '-' + names[now_site]
                        prev_site = 0

    # Save things to sites
    sites = {}
    sites['site_radius'] = dist_close
    sites['frac_pos'] = sites_pos
    sites['cart_pos'] = cart_pos
    sites['site_names'] = names
    sites['occupancy'] = np.sum(site_occup, axis=1)
    sites['occup_parts'] = site_occup
    sites['atoms'] = atom_site
    sites['transitions'] = transitions
    sites['all_trans'] = all_trans
    sites['success'] = succes
    finished = True

    return sites, finished
