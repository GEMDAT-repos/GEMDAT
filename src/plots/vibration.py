import matplotlib.pyplot as plt
import numpy as np
from gemdat.vibration import Vibration
from scipy import stats


def frequency_vs_occurence(*, vibration: Vibration, **kwargs):
    """Plot attempt frequency vs occurence.

    Parameters
    ----------
    vibration : Vibration
        Vibration object which describes vibrations in trajectories
    """
    speed = vibration.speed()
    length = speed.shape[1]
    half_length = length // 2 + 1

    trans = np.fft.fft(speed)

    two_sided = np.abs(trans / length)
    one_sided = two_sided[:, :half_length]

    fig, ax = plt.subplots()

    f = vibration.fs * np.arange(half_length) / length

    sum_freqs = np.sum(one_sided, axis=0)
    smoothed = np.convolve(sum_freqs, np.ones(51), 'same') / 51
    ax.plot(f, smoothed, linewidth=3)

    y_max = np.max(sum_freqs)

    attempt_freq, attempt_freq_std = vibration.attempt_frequency()

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

    return fig


def vibrational_amplitudes(*, vibration: Vibration, **kwargs):
    """Plot histogram of vibrational amplitudes with fitted Gaussian.

    Parameters
    ----------
    vibration : Vibration
        Vibration object which describes vibrations in trajectories
    """

    fig, ax = plt.subplots()
    ax.hist(vibration.amplitudes(), bins=100, density=True)

    x = np.linspace(-2, 2, 100)
    y_gauss = stats.norm.pdf(x, 0, vibration.vibration_amplitude())
    ax.plot(x, y_gauss, 'r')

    ax.set(title='Histogram of vibrational amplitudes with fitted Gaussian',
           xlabel='Amplitude (Angstrom)',
           ylabel='Occurrence (a.u.)')

    return fig
