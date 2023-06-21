import numpy as np


def vibration_properties(traj_coords: np.ndarray, *, time_step: float):
    """Get the attempt frequency and vibration amplitude."""
    frequency = 1 / time_step

    len(traj_coords)

    speed = np.diff(diff_displacements, prepend=0, axis=0)

    freq_mean = np.mean(speed, axis=0)

    attempt_freq = np.mean(freq_mean)
    attempt_freq_std = np.std(freq_mean)

    print(f'{attempt_freq=}')
    print(f'{attempt_freq_std=}')

    trans = np.fft.fft(speed, axis=0)
    two_sided = np.abs(trans / len(speed))
    one_sided = two_sided[:len(speed) // 2 + 1]

    ####

    import matplotlib.pyplot as plt

    f = frequency * np.arange(len(speed) // 2 + 1) / len(speed)

    plt.figure()
    sum_freqs = np.sum(one_sided, axis=1)
    smoothed = np.convolve(sum_freqs, np.ones(51), 'same') / 51
    plt.plot(f, smoothed, linewidth=3)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Occurrence (a.u.)')
    plt.gca().set_yticklabels([])
    plt.plot([attempt_freq, attempt_freq], [0, 1], '-r', linewidth=3)
    plt.plot(
        [attempt_freq - attempt_freq_std, attempt_freq - attempt_freq_std],
        [0, 1],
        ':r',
        linewidth=3)
    plt.plot(
        [attempt_freq + attempt_freq_std, attempt_freq + attempt_freq_std],
        [0, 1],
        ':r',
        linewidth=3)
    plt.ylim([0, np.max(sum_freqs)])
    # plt.xlim([0, 2.5E13])
    plt.show()


if __name__ == '__main__':
    from GEMDAT import calculate_displacements, load_project

    vasp_xml = '/run/media/stef/Scratch/md-analysis-matlab-example/vasprun.xml'

    traj_coords, data = load_project(vasp_xml, diffusing_element='Li')

    # skip first timesteps
    equilibration_steps = 1250

    species = data['species']
    lattice = data['lattice']
    diffusing_element = data['diffusing_element']

    print(species)
    print(lattice)
    print(traj_coords.shape)

    displacements = calculate_displacements(traj_coords,
                                            data['lattice'],
                                            equilibration_steps=1250)

    # grab displacements for diffusing element only
    idx = np.argwhere([e.name == diffusing_element for e in data['species']])
    diff_displacements = displacements[:, idx].squeeze()

    vibration_properties(diff_displacements, time_step=data['time_step'])
