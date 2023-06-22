def tracer_properties():
    pass


if __name__ == '__main__':
    from gemdat import load_project

    vasp_xml = '/run/media/stef/Scratch/md-analysis-matlab-example/vasprun.xml'

    traj_coords, data = load_project(vasp_xml, diffusing_element='Li')

    # skip first timesteps
    equilibration_steps = 1250
    # number of diffusion dimensions
    diffusion_dimensions = 3
    # Ionic charge of the diffusing ion
    z_ion = 1

    # expected values:
    # - tracer_diffusion : 0
    # - tracer_conductivity : 0
    # - particle_density : 2.4557e+28
    # - mol_per_liter : 40.777

    angstrom_to_meter = 1e-10
