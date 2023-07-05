import numpy as np
from scipy import signal


class Vibration:

    @staticmethod
    def calculate_all(data, extras) -> dict:
        """Calculate Vibration properties.

        Parameters
        ----------
        data : SimulationData
            Input simulation data
        extras : SimpleNamespace
            Extra variables

        Returns
        -------
        extras : dict[str, float]
            Dictionary with calculated parameters
        """
        fs = 1 / data.time_step

        speed, attempt_freq, attempt_freq_std = Vibration.attempt_frequency(
            extras.diff_displacements, fs=fs)
        amplitudes, vibration_amplitude = Vibration.amplitude(speed)

        return {
            'speed': speed,
            'attempt_freq': attempt_freq,
            'attempt_freq_std': attempt_freq_std,
            'amplitudes': amplitudes,
            'vibration_amplitude': vibration_amplitude,
            'fs': fs,
        }

    @staticmethod
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

    @staticmethod
    def attempt_frequency(displacements: np.ndarray, fs: float = 1):
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
        speed = np.diff(displacements, prepend=0)

        freq_mean = Vibration.meanfreq(speed, fs=fs)

        attempt_freq = np.mean(freq_mean)
        attempt_freq_std = np.std(freq_mean)

        print(f'{attempt_freq=:g}')
        print(f'{attempt_freq_std=:g}')

        return speed, attempt_freq, attempt_freq_std

    @staticmethod
    def speed(displacements: np.ndarray, fs: float = 1):
        return Vibration.attempt_frequency(displacements, fs)[0]

    @staticmethod
    def attempt_freq(displacements: np.ndarray, fs: float = 1):
        return Vibration.attempt_frequency(displacements, fs)[1]

    @staticmethod
    def attempt_freq_std(displacements: np.ndarray, fs: float = 1):
        return Vibration.attempt_frequency(displacements, fs)[2]

    @staticmethod
    def amplitude(speed: np.ndarray) -> tuple[np.ndarray, float]:
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
        vibration_amp: float = np.std(amplitudes)

        print(f'{mean_vib=:g}')
        print(f'{vibration_amp=:g}')

        return np.asarray(amplitudes), vibration_amp

    @staticmethod
    def amplitudes(speed):
        return Vibration.amplitude(speed)[0]

    @staticmethod
    def vibration_amplitude(speed):
        return Vibration.amplitude(speed)[1]
