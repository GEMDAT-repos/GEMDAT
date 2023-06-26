import numpy as np
from scipy import signal


def calculate_meanfreq(x: np.ndarray, fs: float = 1.0):
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


def calculate_diff_displacements(*,
                                 diffusing_element='Li',
                                 displacements,
                                 species):
    idx = np.argwhere([e.name == diffusing_element for e in species])
    return displacements[idx].squeeze()


def calculate_attempt_frequency(displacements: np.ndarray, fs: float = 1):
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

    freq_mean = calculate_meanfreq(speed, fs=fs)

    attempt_freq = np.mean(freq_mean)
    attempt_freq_std = np.std(freq_mean)

    print(f'{attempt_freq=:g}')
    print(f'{attempt_freq_std=:g}')

    return speed, attempt_freq, attempt_freq_std


def calculate_speed(displacements: np.ndarray, fs: float = 1):
    return calculate_attempt_frequency(displacements, fs)[0]


def calculate_attempt_freq(displacements: np.ndarray, fs: float = 1):
    return calculate_attempt_frequency(displacements, fs)[1]


def calculate_attempt_freq_std(displacements: np.ndarray, fs: float = 1):
    return calculate_attempt_frequency(displacements, fs)[2]


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
    vibration_amp: float = np.std(amplitudes)

    print(f'{mean_vib=:g}')
    print(f'{vibration_amp=:g}')

    return np.asarray(amplitudes), vibration_amp


def calculate_amplitudes(speed):
    return calculate_amplitude(speed)[0]


def calculate_vibration_amplitude(speed):
    return calculate_amplitude(speed)[1]
