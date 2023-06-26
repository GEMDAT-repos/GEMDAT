import numpy as np

from .calc_dist_sqrd_frac import calc_dist_sqrd_frac


def calc_rdfs(sites, sim_data, res, max_dist):
    print(
        'Calculating Radial Distribution Functions for the diffusing element....'
    )
    max_bin = int(np.ceil(max_dist / res))
    print('Maximum distance for RDF:', max_dist, 'Angstrom')
    print('Resolution for RDF:', res, 'Angstrom')

    nr_atoms = sim_data['nr_atoms']
    nr_elem = sim_data['nr_elements']
    lat = sim_data['lattice']

    nr_rdfs = len(sites['names'])
    rdf_names = sites['names']

    elem_struct = np.zeros(nr_elem + 1)
    elem_struct[0] = 1
    for i in range(1, nr_elem + 1):
        elem_struct[i] = elem_struct[i - 1] + sim_data['nr_per_element'][i - 1]

    rdfs = np.zeros((nr_elem, max_bin, nr_rdfs))
    integrated = np.zeros((nr_elem, max_bin, nr_rdfs))
    norm_counter = np.zeros(nr_rdfs)
    all_rdf = np.zeros((nr_elem, max_bin))
    atom_rdf = np.zeros((nr_elem, max_bin))

    diff_elem = 0
    for atom in range(1, nr_atoms + 1):
        if sim_data['diffusing_atom'][atom - 1]:
            diff_elem += 1
            for time in range(1, sim_data['nr_steps'] + 1):
                atom_rdf.fill(0.0)
                pos1 = sim_data['frac_pos'][:, atom - 1, time - 1]
                for i in range(1, nr_atoms + 1):
                    if i != atom:
                        dist = np.sqrt(
                            calc_dist_sqrd_frac(
                                pos1, sim_data['frac_pos'][:, i - 1, time - 1],
                                lat))
                        dist_bin = int(np.ceil(dist / res))
                        if dist_bin <= max_bin:
                            elem = 0
                            j = 0
                            while elem == 0:
                                j += 1
                                if i >= elem_struct[j] and i < elem_struct[j +
                                                                           1]:
                                    elem = j
                            atom_rdf[elem - 1, dist_bin - 1] += 1
                all_rdf += atom_rdf
                if sites['atoms'][time - 1, diff_elem - 1] != 0:
                    site_name = sites['site_names'][
                        sites['atoms'][time - 1, diff_elem - 1] - 1]
                    name = 0
                    not_found = True
                    while not_found:
                        name += 1
                        if site_name == rdf_names[name - 1]:
                            rdfs[:, :, name - 1] += atom_rdf
                            norm_counter[name - 1] += 1
                            not_found = False
            print('*', end='')
    print(' ')

    integrated[:, 1:, :] = np.cumsum(rdfs[:, 1:, :], axis=1)

    rdf = {
        'distributions': rdfs,
        'integrated': integrated,
        'rdf_names': rdf_names,
        'elements': sim_data['elements'],
        'max_dist': max_dist,
        'resolution': res,
        'total': all_rdf
    }

    return rdf
