import matplotlib.pyplot as plt
import numpy as np
from gemdat.simulation_metrics import SimulationMetrics
from gemdat.trajectory import Trajectory
from scipy import stats


def frequency_vs_occurence(*, trajectory: Trajectory) -> plt.Figure:
    """Plot attempt frequency vs occurence.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory

    Returns
    -------
    fig : plt.Figure
        Output figure
    """
    metrics = SimulationMetrics(trajectory)
    speed = metrics.speed()

    length = speed.shape[1]
    half_length = length // 2 + 1

    trans = np.fft.fft(speed)

    two_sided = np.abs(trans / length)
    one_sided = two_sided[:, :half_length]

    fig, ax = plt.subplots()

    f = trajectory.sampling_frequency * np.arange(half_length) / length

    sum_freqs = np.sum(one_sided, axis=0)
    smoothed = np.convolve(sum_freqs, np.ones(51), 'same') / 51
    ax.plot(f, smoothed, linewidth=3)

    y_max = np.max(sum_freqs)

    attempt_freq, attempt_freq_std = metrics.attempt_frequency()

    print(attempt_freq, attempt_freq_std)

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


def vibrational_amplitudes(*, trajectory: Trajectory) -> plt.Figure:
    """Plot histogram of vibrational amplitudes with fitted Gaussian.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory

    Returns
    -------
    fig : plt.Figure
        Output figure
    """
    metrics = SimulationMetrics(trajectory)

    fig, ax = plt.subplots()
    ax.hist(metrics.amplitudes(), bins=100, density=True)

    x = np.linspace(-2, 2, 100)
    y_gauss = stats.norm.pdf(x, 0, metrics.vibration_amplitude())
    ax.plot(x, y_gauss, 'r')

    ax.set(title='Histogram of vibrational amplitudes with fitted Gaussian',
           xlabel='Amplitude (Angstrom)',
           ylabel='Occurrence (a.u.)')

    return fig
