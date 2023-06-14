import timeit

import numpy as np
import scipy.io as sio

from .frac_and_disp import frac_and_disp
from .tracer_properties import tracer_properties
from .vibration_properties import vibration_properties


def read_lammps(xyz_file, lammps_input, lammps_output, lammps_structure,
                equil_time, diff_elem, output_file, diff_dim, z_ion):
    # Define constants
    sim_data = {}
    sim_data['e_charge'] = 1.60217657e-19  # Electron charge
    sim_data['k_boltzmann'] = 1.3806488e-23  # Boltzmann's constant
    sim_data['avogadro'] = 6.022140857e23  # Avogadro's number

    sim_data['diffusion_dim'] = diff_dim
    sim_data['ion_charge'] = z_ion

    # Initialise
    sim_data['lattice'] = np.zeros((3, 3))
    sim_data['diff_elem'] = diff_elem
    sim_data['equilibration_time'] = equil_time

    # Read output of simulation
    sim_data = read_lammpsfiles(xyz_file, lammps_input, lammps_output,
                                lammps_structure, sim_data)

    # Calculate Attempt frequency and standard deviation of vibration distance
    sim_data['attempt_freq'], sim_data['vibration_amp'], sim_data[
        'std_attempt_freq'] = vibration_properties(sim_data, False)

    # Tracer diffusion coefficient and conductivity
    sim_data['tracer_diffusion'], sim_data['tracer_conductivity'], sim_data[
        'particle_density'], sim_data['mol_per_liter'] = tracer_properties(
            sim_data)

    # Save sim_data in a .mat file
    sio.savemat(output_file, {'sim_data': sim_data})

    #print('Finished reading in the OUTCAR file after {} minutes'.format(toc() / 60))


def read_lammpsfiles(xyz_file, lammps_input, lammps_output, lammps_structure,
                     sim_data):
    #tic()

    # First read in things that are constant throughout the MD simulation

    # From the input file of LAMMPS get some settings
    with open(lammps_input, 'r') as file:
        for line in file:
            temp = line.split()
            if len(temp) > 1:
                if temp[0] == 'run':  # The total number of MD steps
                    sim_data['total_steps'] = int(temp[1])
                elif temp[1] == 'T1' and temp[
                        2] == 'equal':  # Temperature of the MD simulation
                    sim_data['temperature'] = float(temp[3])
                elif temp[
                        0] == 'timestep':  # Size of the timestep (*1E-12 = in picoseconds)
                    sim_data['time_step'] = float(temp[1]) * 1E-12

    # From the outputfile of LAMMPS get the (optimized) volume and lattice
    with open(lammps_output, 'r') as file:
        no_tilt = True
        for line in file:
            temp = line.split()
            if len(temp) > 1:
                if temp[1] == 'v_Timer':
                    line_before = temp

                    line = next(file)
                    temp = line.split()
                    sim_data['lattice'][0, 0] = float(temp[7])
                    sim_data['lattice'][1, 1] = float(temp[8])
                    sim_data['lattice'][2, 2] = float(temp[9])

                    if line_before[9] == 'Xy':  # For non-rectangular lattices
                        no_tilt = False
                        sim_data['lattice'][1, 0] = float(temp[10])
                        sim_data['lattice'][2, 0] = float(temp[11])
                        sim_data['lattice'][2, 1] = float(temp[12])
                        sim_data['volume'] = float(
                            temp[13]
                        ) * 1E-30  # Volume of the simulation, in m^3
                    else:  # Assume a rectangular lattice as there is no tilt given
                        sim_data['volume'] = float(
                            temp[10]
                        ) * 1E-30  # Volume of the simulation, in m^3

    if no_tilt:
        print('WARNING! No tilt found in the LAMMPS output')
        print('WARNING! Assuming a rectangular lattice!')

    # From the structure file of LAMMPS
    with open(lammps_structure, 'r') as file:
        for line in file:
            temp = line.split()
            if len(temp) > 1:
                if temp[1] == 'atoms':
                    sim_data['nr_atoms'] = int(
                        temp[0])  # Total number of atoms in the simulation
                    sim_data['atom_element'] = [None] * sim_data['nr_atoms']
                elif len(temp) > 2:
                    if temp[1] == 'atom' and temp[
                            2] == 'types':  # Amount of different elements
                        sim_data['nr_elements'] = int(temp[0])
                        sim_data['nr_per_element'] = np.zeros(
                            sim_data['nr_elements'])
                        sim_data['elements'] = [None] * sim_data['nr_elements']
                elif temp[0] == 'Masses':
                    next(file)  # Skip this line
                    for i in range(sim_data['nr_elements']):
                        line = next(file)
                        temp = line.split()
                        sim_data['element_mass'][i] = float(temp[1])
                        sim_data['elements'][i] = element_from_mass(
                            sim_data['element_mass'][i])

                elif temp[
                        0] == 'Atoms':  # Start counting the number of atoms per element
                    next(file)  # Skip this line
                    for j in range(sim_data['nr_atoms']):
                        line = next(file)
                        temp = line.split()
                        if temp[0] == '':
                            nr = int(
                                temp[3]
                            )  # Sometimes temp starts with '', sometimes it does not
                        else:
                            nr = int(temp[2])
                        sim_data['nr_per_element'][
                            nr] += 1  # Count the number of atoms per element
                        sim_data['atom_element'][j] = sim_data['elements'][
                            nr]  # Remember which element each atom is

    for i in range(sim_data['nr_elements']):
        print('Found {:4d} atoms of element {:3s}'.format(
            sim_data['nr_per_element'][i], sim_data['elements'][i]))

    # Diffusing element specific
    counter = 0
    sim_data['diffusing_atom'] = np.zeros(sim_data['nr_atoms'], dtype=bool)

    sim_data['nr_diffusing'] = 0
    for i in range(sim_data['nr_elements']):
        if sim_data['elements'][i] == sim_data[
                'diff_elem']:  # Where to find the diffusing atoms in sim_data
            sim_data['nr_diffusing'] += sim_data['nr_per_element'][
                i]  # Give the diffusing atoms the value 'True'
            for j in range(counter, counter + sim_data['nr_per_element'][i]):
                sim_data['diffusing_atom'][j] = True
        counter += sim_data['nr_per_element'][i]

    # Check if the given diffusing element is present
    if sim_data['nr_diffusing'] == 0:
        raise ValueError(
            'Given diffusing element not found in inputfile! Check your input')

    # Check and define simulation dependent things
    sim_data['equilibration_steps'] = sim_data[
        'equilibration_time'] / sim_data['time_step']
    # Number of steps to be used for the analysis
    sim_data['nr_steps'] = round(sim_data['total_steps'] -
                                 sim_data['equilibration_steps'])
    print(
        'Throwing away the first {:4.0f} steps because of the chosen equilibration time.'
        .format(sim_data['equilibration_steps']))

    # Now read positions of all atoms during the simulation from the xyz-file
    with open(xyz_file, 'r') as file:
        sim_data['cart_pos'] = np.zeros(
            (3, sim_data['nr_atoms'],
             sim_data['nr_steps']))  # Define cartesian positions array
        nr_atoms = sim_data['nr_atoms']
        pos_line = 'Atoms.'  # After which word the positions of the next timestep begin
        time = 0
        step = 0
        skip_steps = sim_data['equilibration_steps']
        for line in file:
            check_start = line.split()
            if check_start[
                    0] == pos_line:  # Check if the line starts with pos_line
                time += 1
                if time > skip_steps:  # Equilibration steps are thrown away
                    step += 1  # The time step
                    for atom in range(nr_atoms):  # Loop over the atoms
                        line = next(file).split()  # The next line
                        for j in range(3):
                            sim_data['cart_pos'][j, atom,
                                                 step] = float(line[j + 1])
                if (step + 1
                    ) % 2500 == 0:  # Show that reading in is still happening
                    print('Reading timestep {} of {} after {:.2f} minutes.'.
                          format(step + 1, sim_data['nr_steps'],
                                 timeit.default_timer() / 60))

    # Reading positions done
    # If the simulation was not completely finished
    if step != sim_data['nr_steps']:
        sim_data['nr_steps'] = step
        temp_cart = sim_data['cart_pos'][:, :, :step]
        sim_data['cart_pos'] = np.zeros(
            (3, sim_data['nr_atoms'], sim_data['nr_steps']))
        sim_data['cart_pos'] = temp_cart
    # Total simulated time
    sim_data['total_time'] = sim_data['nr_steps'] * sim_data['time_step']

    # Determine fractional positions and displacement
    sim_data = frac_and_disp(sim_data)

    return sim_data


def element_from_mass(mass):
    # Determine the name of the element based on the given mass
    if mass == 1:
        element = 'H'
    elif mass == 7:
        element = 'Li'
    elif mass == 16:
        element = 'O'
    elif mass == 23:
        element = 'Na'
    elif mass == 27:
        element = 'Al'
    elif mass == 31:
        element = 'P'
    elif mass == 32:
        element = 'S'
    else:
        raise ValueError(
            'Element not recognized based on its mass! Add it to element_from_mass in read_lammps.py'
        )
    return element
