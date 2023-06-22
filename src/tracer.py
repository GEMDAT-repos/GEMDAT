import numpy as np


def tracer_properties():
    pass


if __name__ == '__main__':
    from gemdat import calculate_displacements, load_project
    from gemdat.constants import avogadro, e_charge, k_boltzmann

    vasp_xml = '/run/media/stef/Scratch/md-analysis-matlab-example/vasprun.xml'

    traj_coords, data = load_project(vasp_xml, diffusing_element='Li')

    lattice = data['lattice']
    species = data['species']
    diffusing_element = data['diffusing_element']
    time_step = data['time_step']

    # skip first timesteps
    equilibration_steps = 1250
    # number of diffusion dimensions
    dimensions = 3
    # Ionic charge of the diffusing ion
    z_ion = 1

    angstrom_to_meter = 1e-10

    nr_steps = len(traj_coords) - equilibration_steps
    nr_atoms = len(species)

    total_time = nr_steps * time_step

    volume_ang = lattice.volume
    volume_m3 = volume_ang * angstrom_to_meter**3

    nr_diffusing = sum([e.name == diffusing_element for e in species])
    particle_density = nr_diffusing / volume_m3

    mol_per_liter = (particle_density * 1e-3) / avogadro

    print(f'{particle_density=:g} m^-3')
    print(f'{mol_per_liter=:g} mol/l')

    displacements = calculate_displacements(
        traj_coords, lattice, equilibration_steps=equilibration_steps)

    # grab displacements for diffusing element only
    idx = np.argwhere([e.name == diffusing_element for e in species])
    diff_displacements = displacements[idx].squeeze()

    # Matlab code contains a bug here so I'm not entirely sure what is the definition
    # Matlab code takes the first column, which is equal to 0
    # Do they mean the total displacement (i.e. last column)?
    msd = np.mean(diff_displacements[:, -1]**2)  # Angstron^2

    temperature = data['temperature']

    # Diffusivity = MSD/(2*dimensions*time)
    tracer_diff = (msd * angstrom_to_meter**2) / (2 * dimensions * total_time)
    # Conductivity = elementary_charge^2 * charge_ion^2 * diffusivity * particle_density / (k_B * T)
    tracer_conduc = ((e_charge**2) * (z_ion**2) * tracer_diff *
                     particle_density) / (k_boltzmann * temperature)

    print(f'{tracer_diff=:g} m^2/s')
    print(f'{tracer_conduc=:g} S/m')

    # expected values:
    from math import isclose
    # assert tracer_diffusion == 0
    # assert tracer_conductivity == 0
    assert isclose(particle_density, 2.4557e28, rel_tol=1e-4)
    assert isclose(mol_per_liter, 40.777, rel_tol=1e-4)
