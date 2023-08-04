from __future__ import annotations

import typing
from functools import lru_cache

import numpy as np
from pymatgen.core.units import FloatWithUnit
from scipy import signal

if typing.TYPE_CHECKING:

    from gemdat.trajectory import Trajectory


class Vibration:

    def __init__(self, trajectory: Trajectory, fs: float = 1.0):
        """Calculate Vibration properties.

        Parameters
        ----------
        trajectory : Trajectory
            Input trajectory
        fs : float, optional
            Sampling frequency
        """

        self.trajectory = trajectory
        self.fs = 1 / trajectory.time_step

    @staticmethod
    def meanfreq(x: np.ndarray, fs):
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

    def calculate_all(self) -> dict:
        """
        Returns
        -------
        extras : dict[str, float]
            Dictionary with calculated parameters
        """
        speed = Vibration.speed()
        attempt_freq, attempt_freq_std = self.attempt_frequency()
        amplitudes = self.amplitudes()
        vibration_amplitude = self.vibration_amplitude()

        return {
            'speed': speed,
            'attempt_freq': attempt_freq,
            'attempt_freq_std': attempt_freq_std,
            'amplitudes': amplitudes,
            'vibration_amplitude': vibration_amplitude,
            'fs': self.fs,
        }

    @lru_cache
    def speed(self) -> np.ndarray:
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
        return np.diff(self.trajectory.displacements(), prepend=0)

    @lru_cache
    def attempt_frequency(self) -> tuple[float, float]:
        """Calculate attempt frequency.

        Parameters
        ----------
        speed : np.ndarray
            Input array with displacement speeds

        Returns
        -------
        attempt_freq : float
            Attempt frequency
        attempt_freq_std : float
            Attempt frequency standard deviation
        """
        freq_mean = self.meanfreq(self.speed(), fs=self.fs)

        attempt_freq = np.mean(freq_mean)
        attempt_freq_std = np.std(freq_mean)

        attempt_freq = FloatWithUnit(attempt_freq, 'hz')
        attempt_freq_std = FloatWithUnit(attempt_freq_std, 'hz')

        return attempt_freq, attempt_freq_std

    @lru_cache
    def amplitudes(self) -> np.ndarray:
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

        for i, speed_range in enumerate(self.speed()):
            signs = np.sign(speed_range)

            # get indices where sign flips
            splits = np.where(signs != np.roll(signs, shift=-1))[0]
            # strip first and last splits
            subarrays = np.split(speed_range, splits[1:-1] + 1)

            amplitudes.extend([np.sum(array) for array in subarrays])

        return np.asarray(amplitudes)

    def vibration_amplitude(self) -> float:
        mean_vib = np.mean(self.amplitudes())
        vibration_amp: float = np.std(self.amplitudes())

        mean_vib = FloatWithUnit(mean_vib, 'ang')
        vibration_amp = FloatWithUnit(vibration_amp, 'ang')

        return vibration_amp
