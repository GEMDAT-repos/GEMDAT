import numpy as np

from .calc_dist_sqrd_frac import calc_dist_sqrd_frac


def possible_collective(sites, sim_data, coll_dist):
    print('Determining collective jumps')
    coll_steps = np.ceil(
        (1.0 / sim_data['attempt_freq']) / sim_data['time_step'])

    coll_dist_sqrd = coll_dist * coll_dist
    lattice = sim_data['lattice']

    if np.sum(sites['transitions']) > 0:
        times, orig = np.sort(sites['all_trans'][:, 4])
        nr_jumps = len(times)
    else:
        nr_jumps = 0

    coll_count = 0
    uncoll_count = 0
    collective = np.zeros((1, 2))
    coll_jumps = ['', '']
    if nr_jumps > 1:
        for i in range(nr_jumps - 1):
            j = i + 1
            while j <= nr_jumps and (times[j] - times[i] <= coll_steps):
                if sites['all_trans'][orig[i],
                                      0] != sites['all_trans'][orig[j], 0]:
                    start_site_i = sites['all_trans'][orig[i], 1]
                    start_site_j = sites['all_trans'][orig[j], 1]
                    end_site_i = sites['all_trans'][orig[i], 2]
                    end_site_j = sites['all_trans'][orig[j], 2]
                    ss_dist_sqrd = calc_dist_sqrd_frac(
                        sites['frac_pos'][:, start_site_i],
                        sites['frac_pos'][:, start_site_j], lattice)
                    se_dist_sqrd = calc_dist_sqrd_frac(
                        sites['frac_pos'][:, start_site_i],
                        sites['frac_pos'][:, end_site_j], lattice)
                    es_dist_sqrd = calc_dist_sqrd_frac(
                        sites['frac_pos'][:, end_site_i],
                        sites['frac_pos'][:, start_site_j], lattice)
                    ee_dist_sqrd = calc_dist_sqrd_frac(
                        sites['frac_pos'][:, end_site_i],
                        sites['frac_pos'][:, end_site_j], lattice)
                    if ss_dist_sqrd <= coll_dist_sqrd or se_dist_sqrd <= coll_dist_sqrd or \
                       es_dist_sqrd <= coll_dist_sqrd or ee_dist_sqrd <= coll_dist_sqrd:
                        coll_count += 1
                        collective = np.vstack((collective, [orig[i],
                                                             orig[j]]))
                        coll_jumps = np.vstack((coll_jumps, [
                            f'{sites["site_names"][start_site_i]}_to_{sites["site_names"][end_site_i]}',
                            f'{sites["site_names"][start_site_j]}_to_{sites["site_names"][end_site_j]}'
                        ]))
                    else:
                        uncoll_count += 1
                else:
                    uncoll_count += 1
                j += 1

    print(f'Total number of jumps: {nr_jumps}')
    print(f'Number of possibly collective jumps found: {coll_count}')
    print(f'Number of solo jumps found: {uncoll_count}')

    coll_matrix = np.zeros(
        (len(sites['jump_names']), len(sites['jump_names'])))
    if coll_count > 0:
        for a in range(coll_count):
            i = 0
            j = 0
            for b in range(len(sites['jump_names'])):
                if sites['jump_names'][b] == coll_jumps[a, 0]:
                    i = b
                if sites['jump_names'][b] == coll_jumps[a, 1]:
                    j = b
            coll_matrix[i, j] += 1

    multi_coll = 0
    if coll_count > 1:
        uni = np.unique(collective)
        if len(uni) != len(collective):
            print(
                f'Nr. of multiply collective jumps found: {len(collective) - len(uni)}'
            )
            multi_coll = np.zeros(len(collective) - len(uni))
            count = 0
            sorted_indices = np.argsort(
                np.concatenate((collective[:, 0], collective[:, 1])))
            for i in range(1, len(sorted_indices)):
                if sorted_indices[i] == sorted_indices[i - 1]:
                    multi_coll[count] = sorted_indices[i]
                    count += 1

    return collective, coll_jumps, coll_matrix, multi_coll, uncoll_count
