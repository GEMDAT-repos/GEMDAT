import numpy as np


def tracer_properties(sim_data):
    ang2m = 1E-10
    dimensions = sim_data['diffusion_dim']
    z_ion = sim_data['ion_charge']

    print('-------------------------------------------')
    print('Calculating tracer diffusion and conductivity based on:')
    print(
        f'{dimensions} dimensional diffusion, and an ion with a charge of {z_ion}'
    )

    particle_density = sim_data['nr_diffusing'] / sim_data['volume']
    mol_per_liter = particle_density / (1000 * sim_data['avogadro'])

    displacement = []
    for i in range(sim_data['nr_atoms']):
        if sim_data['diffusing_atom'][i]:
            displacement.append(sim_data['displacement'][i])

    displacement = np.array(displacement)
    sqrd_disp = displacement**2
    msd = np.mean(sqrd_disp)

    tracer_diff = (msd * ang2m**2) / (2 * dimensions * sim_data['total_time'])
    tracer_conduc = ((sim_data['e_charge'] ** 2) * (z_ion ** 2) * tracer_diff * particle_density) / \
                    (sim_data['k_boltzmann'] * sim_data['temperature'])

    print(
        f'Tracer diffusivity determined to be (in meter^2/sec): {tracer_diff}')
    print(
        f'Tracer conductivity determined to be (in Siemens/meter): {tracer_conduc}'
    )
    print('-------------------------------------------')

    return tracer_diff, tracer_conduc, particle_density, mol_per_liter
