from __future__ import annotations

import typing

import numpy as np
from pymatgen.core.units import FloatWithUnit
from scipy import signal

if typing.TYPE_CHECKING:
    from types import SimpleNamespace

    from gemdat.trajectory import GemdatTrajectory


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


class Vibration:

    @staticmethod
    def calculate_all(trajectory: GemdatTrajectory,
                      extras: SimpleNamespace) -> dict:
        """Calculate Vibration properties.

        Parameters
        ----------
        trajectory : GemdatTrajectory
            Input trajectory
        extras : SimpleNamespace
            Extra variables

        Returns
        -------
        extras : dict[str, float]
            Dictionary with calculated parameters
        """
        fs = 1 / trajectory.time_step

        speed = Vibration.speed(extras.diff_displacements)
        attempt_freq, attempt_freq_std = Vibration.attempt_frequency(speed,
                                                                     fs=fs)

        amplitudes = Vibration.amplitudes(speed)
        vibration_amplitude = Vibration.vibration_amplitude(amplitudes)

        return {
            'speed': speed,
            'attempt_freq': attempt_freq,
            'attempt_freq_std': attempt_freq_std,
            'amplitudes': amplitudes,
            'vibration_amplitude': vibration_amplitude,
            'fs': fs,
        }

    @staticmethod
    def speed(displacements: np.ndarray) -> np.ndarray:
        """Calculate attempt frequency.

        Parameters
        ----------
        displacements : np.ndarray
            Input array with displacements

        Returns
        -------
        speed : np.ndarray
            Output array with speeds
        """
        return np.diff(displacements, prepend=0)

    @staticmethod
    def attempt_frequency(speed: np.ndarray,
                          fs: float = 1) -> tuple[float, float]:
        """Calculate attempt frequency.

        Parameters
        ----------
        speed : np.ndarray
            Input array with displacement speeds
        fs : float, optional
            Sampling frequency

        Returns
        -------
        attempt_freq : float
            Attempt frequency
        attempt_freq_std : float
            Attempt frequency standard deviation
        """
        freq_mean = meanfreq(speed, fs=fs)

        attempt_freq = np.mean(freq_mean)
        attempt_freq_std = np.std(freq_mean)

        attempt_freq = FloatWithUnit(attempt_freq, 'hz')
        attempt_freq_std = FloatWithUnit(attempt_freq_std, 'hz')

        return attempt_freq, attempt_freq_std

    @staticmethod
    def amplitudes(speed: np.ndarray) -> np.ndarray:
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

        return np.asarray(amplitudes)

    @staticmethod
    def vibration_amplitude(amplitudes: np.ndarray) -> float:
        mean_vib = np.mean(amplitudes)
        vibration_amp: float = np.std(amplitudes)

        mean_vib = FloatWithUnit(mean_vib, 'ang')
        vibration_amp = FloatWithUnit(vibration_amp, 'ang')

        return vibration_amp
