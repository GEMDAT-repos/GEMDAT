import os
import re
import time

import numpy as np
import scipy.io as sio

from .frac_and_disp import frac_and_disp
from .frac_to_cart import frac_to_cart
from .tracer_properties import tracer_properties
from .vibration_properties import vibration_properties


def read_vasp(outcar_file, vasprun_file, equil_time, diff_elem, output_file,
              diff_dim, z_ion):
    sim_data = {}
    sim_data['e_charge'] = 1.60217657e-19
    sim_data['k_boltzmann'] = 1.3806488e-23
    sim_data['avogadro'] = 6.022140857e23

    sim_data['diffusion_dim'] = diff_dim
    sim_data['ion_charge'] = z_ion

    sim_data['lattice'] = np.zeros((3, 3))
    sim_data['diff_elem'] = diff_elem
    sim_data['equilibration_time'] = equil_time

    sim_data = read_outcar(outcar_file, vasprun_file, sim_data)

    sim_data['attempt_freq'], sim_data['vibration_amp'], sim_data[
        'std_attempt_freq'] = vibration_properties(sim_data, False)

    sim_data['tracer_diffusion'], sim_data['tracer_conductivity'], sim_data[
        'particle_density'], sim_data['mol_per_liter'] = tracer_properties(
            sim_data)

    sio.savemat(output_file, {'sim_data': sim_data})

    print(
        'Finished reading in the settings and atomic positions after {} minutes'
        .format(time.time() / 60))


def read_outcar(outcar_file, vasprun_file, sim_data):
    time = 0
    nr_elements = 0

    line_prev = None

    # First read in things that are constant throughout the MD simulation
    with open(outcar_file, 'r') as file:
        line = file.readline()
        while time == 0:
            temp = line.split()
            if len(temp) > 1:
                if temp[1] == 'POSITION' and temp[2] == 'TOTAL-FORCE':
                    time += 1
                elif temp[1] == 'NSW':
                    total_steps = int(temp[3])
                elif temp[1] == 'direct':
                    for i in range(1, 4):
                        line = file.readline()
                        temp = line.split()
                        for j in range(2, 5):
                            sim_data['lattice'][i - 1, j - 2] = float(temp[j])
                elif temp[1] == 'energy' and temp[4] == 'atom':
                    temp_prev = line_prev.split()
                    nr_elements += 1
                    sim_data['elements'][nr_elements - 1, 0] = temp_prev[2]
                elif temp[1] == 'ions':
                    for j in range(6, 5 + nr_elements):
                        per_elem = int(temp[j])
                        sim_data['nr_per_element'][j - 6, 0] = per_elem
                    sim_data['nr_atoms'] = np.sum(sim_data['nr_per_element'])
                elif temp[1] == 'TEBEG':
                    temp = line.split()
                    sim_data['temperature'] = float(temp[4])
                elif temp[1] == 'POTIM':
                    sim_data['time_step'] = float(temp[4]) * 1E-15
                elif temp[1] == 'volume':
                    sim_data['volume'] = float(temp[6]) * 1E-30
                elif temp[1] == 'Mass':
                    line = file.readline()
                    temp = line.split()
                    if len(temp) < 3 + nr_elements:
                        for j in range(4, len(temp)):
                            if len(temp[j]) > 6:
                                correct_mass = re.split(
                                    '(\\d*\\.\\d\\d)', temp[j])[1:]
                                temp_mass = temp + [' ']
                                if len(correct_mass) > 2:
                                    for k in range(3, len(correct_mass)):
                                        temp_mass += [' ']
                                temp_mass[j + 2:] = temp[j + 1:]
                                for k in range(len(correct_mass)):
                                    temp_mass[j + k - 1] = correct_mass[k]
                                temp = temp_mass
                    for j in range(4, 3 + nr_elements):
                        elem_mass = float(temp[j])
                        sim_data['element_mass'][j - 3] = elem_mass
            line = file.readline()

    sim_data['nr_elements'] = nr_elements

    counter = 0
    sim_data['atom_element'] = [''] * sim_data['nr_atoms']
    sim_data['diffusing_atom'] = [False] * sim_data['nr_atoms']

    sim_data['nr_diffusing'] = 0
    for i in range(1, sim_data['nr_elements'] + 1):
        if sim_data['elements'][i - 1, 0] == sim_data['diff_elem']:
            sim_data['nr_diffusing'] += sim_data['nr_per_element'][i - 1, 0]
            for j in range(counter,
                           counter + sim_data['nr_per_element'][i - 1, 0]):
                sim_data['diffusing_atom'][j] = True
        for j in range(counter,
                       counter + sim_data['nr_per_element'][i - 1, 0]):
            sim_data['atom_element'][j] = sim_data['elements'][i - 1, 0]
        counter += sim_data['nr_per_element'][i - 1, 0]

    if sim_data['nr_diffusing'] == 0:
        raise ValueError(
            'ERROR! Given diffusing element not found in input file! Check your input'
        )

    sim_data['equilibration_steps'] = sim_data[
        'equilibration_time'] / sim_data['time_step']

    if np.isnan(total_steps):
        print(
            'WARNING! Total number of steps is undefined in OUTCAR, assuming 1 million steps!'
        )
        print(
            'WARNING! This will be adjusted after the atomic positions have been read in.'
        )
        total_steps = 1000000

    sim_data['nr_steps'] = round(total_steps - sim_data['equilibration_steps'])
    print(
        f'Throwing away the first {sim_data["equilibration_steps"]:4.0f} steps '
        'because of the chosen equilibration time.')

    pos_line = ' POSITION                                       TOTAL-FORCE (eV/Angst)'

    sim_data['cart_pos'] = np.zeros(
        (3, sim_data['nr_atoms'], sim_data['nr_steps']))

    nr_atoms = sim_data['nr_atoms']
    step = 0
    skip_steps = sim_data['equilibration_steps']

    with open(outcar_file, 'r') as file:
        line = file.readline()
        while line:
            if line.strip() == pos_line:
                time += 1
                if time > skip_steps:
                    step += 1
                    file.readline()
                    for atom in range(nr_atoms):
                        line = file.readline().split()
                        for j in range(1, 4):
                            sim_data['cart_pos'][j - 1, atom,
                                                 step - 1] = float(line[j])
                if (step + 1) % 2500 == 0:
                    print(
                        f'Reading timestep {step+1} of {sim_data["nr_steps"]} after xxxx minutes'
                    )
            line = file.readline()

    if step == 0 or time < 0.25 * sim_data['nr_steps']:
        print(
            'WARNING! The OUTCAR-file is missing a lot of atomic positions from the MD simulation!'
        )
        if not os.path.exists(vasprun_file):
            raise FileNotFoundError(
                'ERROR! Put vasprun.xml in the folder to read the atomic positions from that file,'
                ' then run analyse_md again')
        else:
            sim_data, step = read_vasprunxml(vasprun_file, sim_data)

    if step != sim_data['nr_steps']:
        sim_data['nr_steps'] = step
        temp_cart = sim_data['cart_pos'][:, :, :step]
        sim_data['cart_pos'] = np.zeros(
            (3, sim_data['nr_atoms'], sim_data['nr_steps']))
        sim_data['cart_pos'] = temp_cart

    sim_data['total_time'] = sim_data['nr_steps'] * sim_data['time_step']

    sim_data = frac_and_disp(sim_data)

    return sim_data


def read_vasprunxml(vasprun_file, sim_data):
    print(
        'WARNING! The atomic positions during the MD simulation are read from vasprun.xml'
    )

    # Start:
    with open(vasprun_file, 'r') as file:
        pos_line = '   <varray name="positions" >'
        nr_atoms = sim_data['nr_atoms']
        step = 0
        lattice = sim_data['lattice']
        skip_steps = sim_data['equilibration_steps']
        frac_pos = np.zeros(3)
        time = 0

        line = file.readline().strip()  # the first line
        while line:
            if line == pos_line:
                time += 1
                if time > skip_steps:
                    step += 1
                    # Faster way should be possible by reading all coordinates at once instead of using loops,
                    # but this implementation should suffice for now
                    for atom in range(nr_atoms):
                        line = file.readline().split()
                        for j in range(3):
                            frac_pos[j] = float(line[j + 2])
                        # Not very efficient, but easy to implement:
                        cart = frac_to_cart(frac_pos, lattice)
                        sim_data['cart_pos'][:, atom, step - 1] = cart
                if (step +
                        1) % 2500 == 0:  # To see that stuff is still happening
                    print(f'Reading timestep {step} after xxxx seconds.')
            line = file.readline().strip()  # the next line

    return sim_data, step
