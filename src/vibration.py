from typing import Optional

import matplotlib.pyplot as plt
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


def calculate_amplitude(speed: np.ndarray) -> tuple[np.ndarray, float]:
    """Calculate vibration amplitude.

    Parameters
    ----------
    speed : np.ndarray
        Input array with displacement velocities

    Returns
    -------
    amplitudes : np.ndarray
        Output array with vibrational amplitudes
    vibration amplitude : float
        Mean vibration amplitude
    """
    amplitudes = []

    for i, speed_range in enumerate(speed):
        signs = np.sign(speed_range)

        # get indices where sign flips
        splits = np.where(signs != np.roll(signs, shift=-1))[0]
        # strip first and last splits
        subarrays = np.split(speed_range, splits[1:-1] + 1)

        amplitudes.extend([np.sum(array) for array in subarrays])

    mean_vib = np.mean(amplitudes)
    vibration_amp = np.std(amplitudes)

    print(f'{mean_vib=:g}')
    print(f'{vibration_amp=:g}')

    return amplitudes, vibration_amp


def calculate_attempt_frequency(displacements: np.ndarray, *, fs: float = 1):
    """Calculate attempt frequency.

    Parameters
    ----------
    displacements : np.ndarray
        Input array with displacements
    fs : float, optional
        Sampling frequency

    Returns
    -------
    speed : np.ndarray
        Output array with speeds
    attempt_freq : float
        Attempt frequency
    attempt_freq_std : float
        Attempt frequency standard deviation
    """
    speed = np.diff(displacements.T, prepend=0)

    freq_mean = meanfreq(speed, fs=fs)

    attempt_freq = np.mean(freq_mean)
    attempt_freq_std = np.std(freq_mean)

    print(f'{attempt_freq=:g}')
    print(f'{attempt_freq_std=:g}')

    return speed, attempt_freq, attempt_freq_std


def plot_frequency_vs_occurence(speed: np.ndarray,
                                *,
                                fs: float = 1,
                                freq: Optional[float] = None,
                                freq_std: Optional[float] = None):
    """Plot attempt frequency vs occurence.

    Parameters
    ----------
    speed : np.ndarray
        Input array with displacement velocities
    fs : float, optional
        Sampling frequency
    freq : Optional[float], optional
        Attempt frequency
    freq_std : Optional[float], optional
        Attempt frequency standard deviation
    """
    length = speed.shape[1]
    half_length = length // 2 + 1

    trans = np.fft.fft(speed)

    two_sided = np.abs(trans / length)
    one_sided = two_sided[:, :half_length]

    fig, ax = plt.subplots()

    f = fs * np.arange(half_length) / length

    sum_freqs = np.sum(one_sided, axis=0)
    smoothed = np.convolve(sum_freqs, np.ones(51), 'same') / 51
    ax.plot(f, smoothed, linewidth=3)

    y_max = np.max(sum_freqs)

    if freq:
        ax.vlines([freq], 0, y_max, colors='red')
    if freq and freq_std:
        ax.vlines([freq + freq_std, freq - freq_std],
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


def plot_vibrational_amplitudes(amplitudes: np.ndarray, *,
                                vibration_amplitude: float):
    """Plot histogram of vibrational amplitudes with fitted Gaussian.

    Parameters
    ----------
    amplitudes : np.ndarray
        Input array with vibrational amplitudes
    vibration_amplitude : float
        Sigma of the vibrational amplitudes
    """
    fig, ax = plt.subplots()
    ax.hist(amplitudes, bins=100, density=True)

    x = np.linspace(-2, 2, 100)
    y_gauss = stats.norm.pdf(x, 0, vibration_amplitude)
    ax.plot(x, y_gauss, 'r')

    ax.set(title='Histogram of vibrational amplitudes with fitted Gaussian',
           xlabel='Amplitude (Angstrom)',
           ylabel='Occurrence (a.u.)')

    plt.show()


if __name__ == '__main__':
    from gemdat import calculate_displacements, load_project

    vasp_xml = '/run/media/stef/Scratch/md-analysis-matlab-example/vasprun.xml'

    traj_coords, data = load_project(vasp_xml, diffusing_element='Li')

    species = data['species']
    lattice = data['lattice']
    diffusing_element = data['diffusing_element']
    time_step = data['time_step']

    displacements = calculate_displacements(traj_coords,
                                            lattice,
                                            equilibration_steps=1250)

    # grab displacements for diffusing element only
    idx = np.argwhere([e.name == diffusing_element for e in species])
    diff_displacements = displacements[:, idx].squeeze()

    fs = 1 / time_step

    speed, attempt_freq, attempt_freq_std = calculate_attempt_frequency(
        diff_displacements, fs=fs)

    plot_frequency_vs_occurence(speed,
                                fs=fs,
                                freq=attempt_freq,
                                freq_std=attempt_freq_std)

    amplitudes, vibration_amplitude = calculate_amplitude(speed)

    plot_vibrational_amplitudes(amplitudes,
                                vibration_amplitude=vibration_amplitude)
