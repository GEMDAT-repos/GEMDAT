def tracer_properties():
    pass


if __name__ == '__main__':
    from gemdat import load_project
    from gemdat.constants import avogadro

    vasp_xml = '/run/media/stef/Scratch/md-analysis-matlab-example/vasprun.xml'

    traj_coords, data = load_project(vasp_xml, diffusing_element='Li')

    lattice = data['lattice']
    species = data['species']
    diffusing_element = data['diffusing_element']

    # skip first timesteps
    equilibration_steps = 1250
    # number of diffusion dimensions
    diffusion_dimensions = 3
    # Ionic charge of the diffusing ion
    z_ion = 1

    angstrom_to_meter = 1e-10

    volume_ang = lattice.volume
    volume_m3 = volume_ang * angstrom_to_meter**3

    nr_diffusing = sum([e.name == diffusing_element for e in species])
    particle_density = nr_diffusing / volume_m3

    mol_per_liter = particle_density / (1000 * avogadro)

    print(f'{particle_density=:g} m^-3')
    print(f'{mol_per_liter=:g} mol/l')

    # expected values:
    # - tracer_diffusion : 0
    # - tracer_conductivity : 0
    # - particle_density : 2.4557e+28
    # - mol_per_liter : 40.777

    breakpoint()
