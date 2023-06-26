from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from ..data import Data


def plot_frequency_vs_occurence(*,
                                data: Data,
                                fs: Optional[float] = None,
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
    if not fs:
        fs = 1 / data.time_step

    length = data.speed.shape[1]
    half_length = length // 2 + 1

    trans = np.fft.fft(data.speed)

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


def plot_vibrational_amplitudes(*,
                                data: Data,
                                vibration_amplitude: Optional[float] = None):
    """Plot histogram of vibrational amplitudes with fitted Gaussian.

    Parameters
    ----------
    amplitudes : np.ndarray
        Input array with vibrational amplitudes
    vibration_amplitude : float
        Sigma of the vibrational amplitudes
    """

    if not vibration_amplitude:
        vibration_amplitude = data.vibration_amplitude

    fig, ax = plt.subplots()
    ax.hist(data.amplitudes, bins=100, density=True)

    x = np.linspace(-2, 2, 100)
    y_gauss = stats.norm.pdf(x, 0, vibration_amplitude)
    ax.plot(x, y_gauss, 'r')

    ax.set(title='Histogram of vibrational amplitudes with fitted Gaussian',
           xlabel='Amplitude (Angstrom)',
           ylabel='Occurrence (a.u.)')

    plt.show()
