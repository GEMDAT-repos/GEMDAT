import numpy as np
from gemdat import SimulationData

VASP_XML = '/run/media/stef/Scratch/md-analysis-matlab-example/vasprun.xml'


def test_tracer():
    equilibration_steps = 1250
    diffusing_element = 'Li'
    dimensions = 3
    z_ion = 1

    data = SimulationData.from_vasprun(VASP_XML, cache='vasprun.xml.cache')

    data.calculate_all(
        equilibration_steps=equilibration_steps,
        diffusing_element=diffusing_element,
        z_ion=z_ion,
        dimensions=dimensions,
    )

    from math import isclose

    assert isclose(data.extras['particle_density'], 2.4557e28, rel_tol=1e-4)
    assert isclose(data.extras['mol_per_liter'], 40.777, rel_tol=1e-4)
    assert isclose(data.extras['tracer_diff'], 1.3524e-09, rel_tol=1e-4)
    assert isclose(data.extras['tracer_conduc'], 94.995, rel_tol=1e-4)


def test_sites():
    from gemdat.io import load_known_material
    from gemdat.sites import (
        calculate_atom_sites,
        calculate_occupancy,
        calculate_transitions,
        calculate_transitions_matrix,
        split_transitions_in_parts,
    )

    equilibration_steps = 1250
    diffusing_element = 'Li'
    n_parts = 10

    data = SimulationData.from_vasprun(VASP_XML, cache='vasprun.xml.cache')

    data.calculate_all(
        equilibration_steps=equilibration_steps,
        diffusing_element=equilibration_steps,
    )

    dist_close = 2 * data.extras['vibration_amplitude']

    structure = load_known_material('argyrodite')

    site_coords = structure.frac_coords

    pdist = data.lattice.get_all_distances(site_coords, site_coords)
    min_dist = np.min(pdist[np.triu_indices_from(pdist, k=1)])

    if min_dist < 2 * dist_close:
        # Crystallographic sites are overlapping with the chosen dist_close, making it smaller
        dist_close = (0.5 * min_dist) - 0.005

        # Two crystallographic sites are within half an Angstrom of each other
        # This is NOT realistic, check/change the given crystallographic site
        if dist_close * 2 < 0.5:
            idx = np.argwhere(pdist == min_dist)
            [f'{structure.sites[i]}-{structure.sites[j]}' for i, j in idx]

            lines = []

            for i, j in idx:
                structure.sites[i]
                site_j = structure.sites[j]
                lines.append('\nToo close:')
                lines.append(
                    '{site_i.specie.name}({i}) {site_i.frac_coords} - ')
                lines.append(f'{site_j.specie.name}({j}) {site_j.frac_coords}')

            msg = ''.join(lines)

            raise ValueError(
                f'Crystallographic sites are too close together (expected: >{dist_close*2:.4f}, '
                f'got: {min_dist:.4f} for {msg}')

    traj_coords = data.trajectory_coords

    species = data.species
    diffusing_idx = np.argwhere([e.name == diffusing_element
                                 for e in species]).flatten()

    diff_coords = traj_coords[equilibration_steps:, diffusing_idx, :]

    assert diff_coords.shape == (73750, 48, 3)

    atom_sites = calculate_atom_sites(coords=diff_coords,
                                      site_coords=site_coords,
                                      lattice=data.lattice,
                                      dist_close=dist_close)

    assert atom_sites.shape == (73750, 48)
    assert atom_sites.sum() == 9228360

    all_transitions = calculate_transitions(atom_sites=atom_sites)

    assert all_transitions.shape == (1336, 5)

    n_diffusing = len(diffusing_idx)

    transitions = calculate_transitions_matrix(all_transitions,
                                               n_diffusing=n_diffusing)

    assert transitions.shape == (48, 48)

    n_steps = len(diff_coords)
    split_transitions = split_transitions_in_parts(all_transitions, n_steps,
                                                   n_parts)
    success = np.stack([
        calculate_transitions_matrix(part, n_diffusing=n_diffusing)
        for part in split_transitions
    ])

    assert len(split_transitions) == n_parts
    assert success.shape == (n_parts, 48, 48)
    assert np.sum(success[0]) == 134
    assert np.sum(success[9]) == 142

    occupancy = calculate_occupancy(atom_sites)

    assert occupancy[-1] == 3015185
    assert occupancy[0] == 1706
    assert occupancy[43] == 6350

    split_atom_sites = np.split(atom_sites, n_parts)
    occupancy_parts = [calculate_occupancy(part) for part in split_atom_sites]

    assert len(occupancy_parts) == n_parts

    assert occupancy_parts[0][0] == 241
    assert occupancy_parts[0][43] == 1231
    assert occupancy_parts[9][0] == 87
    assert occupancy_parts[9][43] == 391
