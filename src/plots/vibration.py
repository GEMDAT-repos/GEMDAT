from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_frequency_vs_occurence(*,
                                speed: np.ndarray,
                                fs: float,
                                attempt_freq: Optional[float] = None,
                                attempt_freq_std: Optional[float] = None,
                                **kwargs):
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

    if attempt_freq:
        ax.vlines([attempt_freq], 0, y_max, colors='red')
    if attempt_freq and attempt_freq_std:
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

    return fig


def plot_vibrational_amplitudes(*, amplitudes: np.ndarray,
                                vibration_amplitude: float, **kwargs):
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

    return fig
