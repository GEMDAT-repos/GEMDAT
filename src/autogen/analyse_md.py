import os

import numpy as np
from scipy.io import loadmat, savemat

from .calc_rates import calc_rates
from .calc_rdfs import calc_rdfs
from .calc_site_occups import calc_site_occups
from .find_sites import find_sites
from .jumps_vs_dist import jumps_vs_dist
from .make_movie import make_movie
from .plot_collective import plot_collective
from .plot_from_displacement import plots_from_displacement
from .plot_rdfs import plot_rdfs
from .plot_sites import plot_jump_paths
from .possible_collective import possible_collective
from .read_lammps import read_lammps
from .read_vasp import read_vasp
from .vibration_properties import vibration_properties


def analyse_md(folder, diff_elem, material):
    # Materials known: 'argyrodite', 'latp', 'LiSnPS', 'na3ps4', 'li3ps4_beta' and 'MnO2_lambda'

    # When you use this code in academic work please cite the accompanying paper:
    # Analysis of diffusion in solid state electrolytes through MD-simulations,
    # improvement of the Li-ion conductivity in \ce{\beta-Li3PS4} as an example;
    # Niek J.J. de Klerk, Eveline van der Maas and Marnix Wagemaker
    # ACS Applied Energy Materials, (2018), doi: 10.1021/acsaem.8b00457

    # Version: 1.3

    ## Settings:
    # !!! WARNING! The settings for equil_time, diffusion_dimensions, z_ion,
    # !!! nr_parts and dist_collective are ONLY used when creating the sim_data.mat and sites.mat files !!!
    # !!! RENAME OR REMOVE THESE FILES IF YOU CHANGE THESE SETTINGS !!!
    equil_time = 2.5E-12  # Equilibration time in seconds (1E-12 = 1 picosecond)
    diffusion_dimensions = 3  # number of diffusion dimensions
    z_ion = 1.0  # Ionic charge of the diffusing ion
    nr_parts = 10  # In how many parts to divide your simulation for statistics:
    dist_collective = 4.5  # Maximum distance for collective motions in Angstrom

    # Pictures
    show_pics = True  # Show plots (or not)
    density_resolution = 0.2  # Resolution for the diffusing atom density plot, in Angstrom
    jump_res = 0.1  # Resolution for the nr. of jumps vs. distance plot, in Angstrom

    # Radial Distribution Functions
    # !!! WARNING! rdf_res and rdf_max_dist are ONLY used when creating the rdfs.mat file !!!
    # !!! RENAME OR REMOVE THE RDF-FILE IF YOU WANT TO CHANGE THESE SETTINGS !!!
    rdfs = False  # Calculate and show Radial Distribution Functions
    rdf_res = 0.1  # Resolution of the RDF bins in Angstrom
    rdf_max_dist = 10  # Maximal distance of the RDF in Angstrom

    # Movie showing the jumps:
    movie = True  # Make a movie showing the jumps (or not)
    nr_steps_frame = 5  # How many time steps per frame in the movie,
    # increase to get smaller and quicker movies
    start_end = [
        5000, 7500
    ]  # The time-steps for which to make a movie ([start at step; end at step])

    ## Standard file names:
    # For the outputs:
    sim_data_file = os.path.join(folder, 'simulation_data.mat')
    sites_file = os.path.join(folder, 'sites.mat')
    rdf_file = os.path.join(folder, 'rdfs.mat')
    movie_file = os.path.join(folder, 'jumps_movie.mp4')

    # For VASP:
    outcar_file = os.path.join(folder, 'OUTCAR')
    vasprun_file = os.path.join(
        folder, 'vasprun.xml'
    )  # Backup for if the atomic positions are not in the OUTCAR

    # For LAMMPS files:
    lammps_input = os.path.join(folder, 'in.500')
    lammps_output = os.path.join(folder, 'out.500')
    lammps_structure = os.path.join(folder, 'structure.dat')
    lammps_xyz = os.path.join(folder, 'dynamics.xyz')

    ## Get simulation data:
    print('Folder given is:', folder)
    if not os.path.isdir(folder):  # Check if given folder exists
        print('ERROR! Given folder does not exist!')
        return

    # Check if simulation_data.mat exists in the given folder:
    if not os.path.isfile(sim_data_file):
        print(sim_data_file, 'not found.')
        # Check for OUTCAR file:
        if os.path.isfile(outcar_file):
            # Read in the simulation data from outcar_file
            print('Reading VASP simulation data from', outcar_file,
                  'this can take a while....')
            sim_data = read_vasp(outcar_file, vasprun_file, equil_time,
                                 diff_elem, sim_data_file,
                                 diffusion_dimensions, z_ion)
        elif os.path.isfile(lammps_xyz):
            print('Reading LAMMPS simulation data from', lammps_xyz,
                  'and some other files, this can take a while....')
            sim_data = read_lammps(lammps_xyz, lammps_input, lammps_output,
                                   lammps_structure, equil_time, diff_elem,
                                   sim_data_file, diffusion_dimensions, z_ion)
        else:
            print(
                'ERROR! File containing MD-output not found in given folder!')
            return
    else:  # sim_data exists already
        print('Found simulation data file in given folder')
        sim_data_mat = loadmat(sim_data_file, matlab_compatible=True)
        sim_data = sim_data_mat['sim_data']

    ## Find sites and transitions
    # Check if sites_file already exists:
    if not os.path.isfile(sites_file):
        # Find the sites if the sites_file does not exist:
        sites, finished = find_sites(sim_data, material, nr_parts)
        if not finished:
            print(
                'Find sites exited with an error, not analyzing this simulation further...'
            )
            return
        # Save the material
        sites.material = material
        # Determine the fractional occupancies of sites:
        (sites.stable_names, sites.sites_occup, sites.atom_locations,
         sites.occup_parts, sites.atom_loc_parts) = calc_site_occups(sites)
        # Calculate jump rates and activation energy for all the differently named
        # transitions between the given sites:
        (sites.jump_names, sites.nr_jumps, sites.rates,
         sites.e_act) = calc_rates(sites, sim_data)
        # Possible correlations between jumps
        (sites.collective, sites.coll_jumps, sites.coll_matrix,
         sites.multi_coll,
         sites.solo_jumps) = possible_collective(sites, sim_data,
                                                 dist_collective)
        # The fraction of solo jumps:
        sites.solo_frac = sites.solo_jumps / np.sum(sites.nr_jumps)
        # The jump diffusivity:
        sites.jump_diffusivity = jumps_vs_dist(sites, sim_data, False,
                                               jump_res)
        # The correlation factor:
        sites.correlation_factor = sim_data.tracer_diffusion / sites.jump_diffusivity
        savemat(sites_file, 'sites')
    else:
        sites_mat = loadmat(sites_file)
        sites = sites_mat['sites']
        print('Found sites file:', sites_file)
        if sites['material'] != material:
            print('ERROR! The material in sites.mat (', sites['material'],
                  ') differs from the given one (', material, ')!')
            print(
                'If this is not a mistake rename or remove sites.mat and try again.'
            )
            return

    ## Show plots:
    if show_pics:
        # Plots based on atomic displacement/position:
        # Displacement per element, for diffusing element per atom, histogram, and density plot:
        plots_from_displacement(sim_data, sites, density_resolution)
        # Jump sites:
        plot_jump_paths(
            sites, sim_data.lattice,
            True)  # True is to show names of sites instead of numbers
        # Nr. of jumps vs. jump distance:
        jumps_vs_dist(sites, sim_data, True, jump_res)
        # Attempt frequency and vibration amplitude:
        vibration_properties(sim_data, True)  # True is to show the figures
        # Collective motions:
        if sites.nr_jumps > 1:
            plot_collective(sites)

    ## Radial distribution functions
    # Check if rdfs == True, and if the file does not exist yet:
    if rdfs and not os.path.isfile(rdf_file):
        print('Radial Distribution Functions file', rdf_file, 'not found')
        rdf = calc_rdfs(sites, sim_data, rdf_res, rdf_max_dist)
        savemat(rdf_file, 'rdf')
        plot_rdfs(rdf)
    elif rdfs:
        print('Using Radial Distribution Functions file', rdf_file)
        loadmat(rdf_file)
        plot_rdfs(rdf)

    ## Movie of the transitions:
    if movie and np.sum(
            sites.nr_jumps) > 0:  # 0 jumps will make a very boring movie
        if not os.path.isfile(movie_file):
            make_movie(sites, sim_data, start_end, movie_file, nr_steps_frame)
        elif os.path.isfile(movie_file):
            print(
                'Movie file', movie_file,
                'found, rename the movie file if you want to make another movie.'
            )
    elif movie and np.sum(
            sites.nr_jumps) == 0:  # 0 jumps will make a very boring movie
        print(
            'Not making a movie because there are no jumps occurring in the MD simulation.'
        )

    print('Analysis of MD simulation done')
