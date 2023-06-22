import os

import matplotlib.pyplot as plt
import numpy as np


def compare_sims(folder):
    temperatures = ['450K', '600K', '750K']

    # Read all subfolders in 'folder':
    subfolders = [
        name for name in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, name))
    ]

    compare_file = os.path.join(folder, 'sims_compare.npy')
    # Load or construct the comparison file:
    if not os.path.exists(compare_file):
        print('sims_compare.npy not found, reading the data from other files')
        sims_comp = read_in_sims(folder, subfolders, temperatures)
        np.save(compare_file, sims_comp)
    else:
        print('sims_compare found')
        sims_comp = np.load(compare_file, allow_pickle=True).item()

    # Plot the results in the compare_file:
    plot_comparison(sims_comp)


def plot_comparison(sims_comp):
    # Properties with 1 value per simulation:
    props_to_plot = [
        'vibration_amp', 'attempt_freq', 'tracer_diffusion',
        'tracer_conductivity', 'total_occup', 'frac_collective',
        'jump_diffusion', 'correlation_factor'
    ]
    titles_of_plots = [
        'Vibration amplitude (Angstrom)', 'Attempt frequency (Hz)',
        'Tracer diffusivity (m^2/sec)', 'Tracer conductivity (S/m)',
        'Known site occupation (%)', 'Collective jumps (%)',
        'Jump diffusivity (m^2/sec)', 'Correlation factor'
    ]

    # Properties with a value per type of jump:
    multi_props_to_plot = ['e_act', 'rates']
    multi_titles_of_plots = ['Activation energy (eV)', 'Jump rate (Hz)']

    linestyles = ['-o', '-^', '-*', '-p', '-+', '-d', '-v', '-<', '->']
    pointstyles = [
        '+', 'o', '*', '+', 'o', '*', '+', 'o', '*', '+', 'o', '*', '+', 'o',
        '*', '+', 'o', '*', '+', 'o', '*', '+'
    ]

    sims = list(sims_comp.keys())
    all_names = []
    for i in range(len(sims)):
        temp = sims[i].split('_')
        all_names.append(temp[1])
    # Find the same names to plot those with a line between them:
    names = np.unique(all_names)

    # Plot properties with 1 value per simulation versus Temperature
    for a in range(len(props_to_plot)):
        plt.figure()
        ax = plt.gca()
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel(titles_of_plots[a])
        for i in range(len(names)):
            counter = 1
            temp_x = []
            temp_y = []
            for j in range(len(all_names)):
                if names[i] == all_names[j]:  # same names
                    temp_x.append(sims_comp[sims[j]]['temperature'])
                    temp_y.append(sims_comp[sims[j]][props_to_plot[a]])
                    counter += 1
            plt.plot(temp_x,
                     temp_y,
                     linestyles[i],
                     linewidth=2.0,
                     markersize=10.0)
        ax.legend(names)
        if props_to_plot[a] in [
                'tracer_diffusion', 'tracer_conductivity', 'jump_diffusion'
        ]:
            ax.set_yscale('log')
        plt.show()

    # Properties with a value per type of jump, plot versus jump name
    for a in range(len(multi_props_to_plot)):
        plt.figure()
        ax = plt.gca()
        ax.set_ylabel(multi_titles_of_plots[a])
        temp_x = np.arange(1, len(sims_comp[sims[0]]['jump_names']) + 1)
        for i in range(len(sims)):
            temp_y = sims_comp[sims[i]][multi_props_to_plot[a]][:, 0]
            plt.plot(temp_x,
                     temp_y,
                     pointstyles[i],
                     linewidth=2.0,
                     markersize=10.0)
        ax.set_xticks(temp_x)
        ax.set_xticklabels(sims_comp[sims[0]]['jump_names'])
        plt.xticks(rotation=90)
        if multi_props_to_plot[a] == 'rates':
            ax.set_yscale('log')
        plt.grid(True)
        plt.show()


def read_in_sims(upfolder, subs, temperatures):
    sims_comp = {}
    for i in range(len(subs)):
        for j in range(len(temperatures)):
            folder = os.path.join(upfolder, subs[i], temperatures[j])
            sim_data_file = os.path.join(folder, 'simulation_data.npy')
            sites_file = os.path.join(folder, 'sites.npy')
            if os.path.exists(sim_data_file) and os.path.exists(sites_file):
                print(
                    f'Loading simulation_data.npy and sites.npy from {folder}')
                sim_data = np.load(sim_data_file, allow_pickle=True).item()
                sites = np.load(sites_file, allow_pickle=True).item()
                # The info from this simulation:
                info = read_sim_info(sim_data, sites)
                # The 'name' of the simulation:
                sim_name = folder.replace('/', '_').replace('.', '')
                sims_comp[sim_name] = info
            elif os.path.exists(sim_data_file):
                print(f'sites.npy not found in given folder: {folder}')
            else:
                print(
                    f'simulation_data.npy and sites.npy not found in given folder: {folder}'
                )
    return sims_comp


def read_sim_info(sim_data, sites):
    # The information to be compared for the simulations:
    # All the info wanted from sim_data:
    info = {}
    info['temperature'] = sim_data['temperature']
    info['attempt_freq'] = sim_data['attempt_freq']
    info['attempt_freq_std'] = sim_data['std_attempt_freq']
    info['vibration_amp'] = sim_data['vibration_amp']
    info['tracer_diffusion'] = sim_data['tracer_diffusion']
    info['tracer_conductivity'] = sim_data['tracer_conductivity']

    # All the info wanted from sites:
    # Fraction of time the atom is at known locations:
    info['total_occup'] = 100 * np.sum(sites['occupancy']) / np.size(
        sites['atoms'])
    info['site_occup'] = 100. * sites['sites_occup']
    info['atom_locations'] = 100. * sites['atom_locations']
    info['site_names'] = sites['stable_names']
    # Jump names:
    info['jump_names'] = sites['jump_names']
    # Activation energy
    info['e_act'] = sites['e_act']
    # Jump rates:
    info['rates'] = sites['rates']
    info['jump_diffusion'] = sites['jump_diffusivity']
    info['correlation_factor'] = sites['correlation_factor']
    # Fraction of collective jumps
    info['frac_collective'] = 100 * (1.0 - sites['solo_frac'])
    return info
