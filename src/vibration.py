import numpy as np
from scipy import signal, stats


def meanfreq(x: np.ndarray, fs: float = 1.0):
    """Estimates the mean frequency in terms of the sample rate, fs.

    Vectorized version of https://stackoverflow.com/a/56487241

    Parameters
    ----------
    x : np.ndarray[i, j]
        Time series of measurement values. The mean frequency is computed
        along the last axis (-1).
    fs : float, optional
        Sampling frequency of the `x` time series. Defaults to 1.0.

    Returns
    -------
    mnfreq : np.ndarray
        Array of mean frequencies.
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)

    assert x.ndim == 2

    f, Pxx_den = signal.periodogram(x, fs, axis=-1)
    width = np.tile(f[1] - f[0], Pxx_den.shape)
    P = Pxx_den * width
    pwr = np.sum(P, axis=1).reshape(-1, 1)

    f = f.reshape(1, -1)

    mnfreq = np.dot(P, f.T) / pwr

    return mnfreq


def vibration_properties(displacements: np.ndarray, *, time_step: float):
    """Get the attempt frequency and vibration amplitude."""
    frequency = 1 / time_step

    length = len(displacements)
    half_length = length // 2 + 1

    speed = np.diff(displacements.T, prepend=0)

    freq_mean = meanfreq(speed, fs=frequency)

    attempt_freq = np.mean(freq_mean)
    attempt_freq_std = np.std(freq_mean)

    print(f'{attempt_freq=:g}')
    print(f'{attempt_freq_std=:g}')

    trans = np.fft.fft(speed)

    two_sided = np.abs(trans / length)
    one_sided = two_sided[:, :half_length]

    ####

    import matplotlib.pyplot as plt

    f = frequency * np.arange(half_length) / length

    fig, ax = plt.subplots()

    sum_freqs = np.sum(one_sided, axis=0)
    smoothed = np.convolve(sum_freqs, np.ones(51), 'same') / 51
    ax.plot(f, smoothed, linewidth=3)

    y_max = np.max(sum_freqs)

    ax.vlines([attempt_freq], 0, y_max, colors='red')
    ax.vlines(
        [attempt_freq + attempt_freq_std, attempt_freq - attempt_freq_std],
        0,
        y_max,
        colors='red',
        linestyles='dashed')

    ax.set(title='Frequency vs Occurence',
           xlabel='Frequency (Hz)',
           ylabel='Occurrence (a.u.)')

    ax.set_ylim([0, y_max])
    ax.set_xlim([-0.1e13, 2.5e13])

    plt.show()

    ###

    amplitude = []

    for i, speed_range in enumerate(speed):
        signs = np.sign(speed_range)

        # get indices where sign flips
        splits = np.where(signs != np.roll(signs, shift=-1))[0]
        # strip first and last splits
        subarrays = np.split(speed_range, splits[1:-1] + 1)

        amplitude.extend([np.sum(array) for array in subarrays])

    mean_vib = np.mean(amplitude)
    vibration_amp = np.std(amplitude)

    print(f'{mean_vib=:g}')
    print(f'{vibration_amp=:g}')

    ###

    fig, ax = plt.subplots()
    ax.hist(amplitude, bins=100, density=True)

    x = np.linspace(-2, 2, 100)
    y_gauss = stats.norm.pdf(x, 0, vibration_amp)
    ax.plot(x, y_gauss, 'r')

    ax.set(title='Histogram of vibrational amplitudes with fitted Gaussian',
           xlabel='Amplitude (Angstrom)',
           ylabel='Occurrence (a.u.)')

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
