import pickle
from functools import cache, cached_property
from pathlib import Path
from typing import Optional

import numpy as np
from pymatgen.io import vasp
from scipy import signal


class Data():
    equilibration_steps = 1250

    def __init__(self, xml_file: Path, cache: Optional[Path] = None):
        if cache and cache.exists():
            with open(cache, 'rb') as f:
                data = pickle.load(f)
            for key, val in data.items():
                setattr(self, key, val)
        else:
            run = vasp.Vasprun(
                xml_file,
                parse_dos=False,
                parse_eigen=False,
                parse_projected_eigen=False,
                parse_potcar_file=False,
            )
            self.structure = run.structures[0]
            trajectory = run.get_trajectory()
            trajectory.to_positions()
            self.trajectory_coords = trajectory.coords
            self.species = self.structure.species
            self.lattice = self.structure.lattice
            self.time_step = run.parameters['POTIM'] * 1e-15
            if cache:
                dump = {
                    'structure': self.structure,
                    'trajectory_coords': self.trajectory_coords,
                    'species': self.species,
                    'lattice': self.lattice,
                    'time_step': self.time_step,
                }
                with open(cache, 'wb') as f:
                    pickle.dump(dump, f)

    @cached_property
    def cell_offsets(self) -> np.ndarray:
        """Calculate cell offsets from trajectory."""
        coords = self.trajectory_coords
        return self.cell_offsets_from_coords(coords)

    def cell_offsets_from_coords(self, coords: np.ndarray) -> np.ndarray:
        """Calculate cell offsets from starting position.

        For example, if a site is at [0, 0, 0.9] -> [0, 0, 0.1]
        assume it has jumped to the next cell: [0, 0, 1.1]

        Parameters
        ----------
        coords : np.ndarray[i, j, k]
            3-dimensional numpy array with dimensions i: time_steps, j: sites, k: coordinates

        Returns
        -------
        offsets : np.ndarray[i, j, k]
            Integer array with unit cell offset vectors.
        """
        first = coords[0, np.newaxis]
        diff = np.diff(coords, axis=0, prepend=first)

        digits = np.digitize(diff, bins=[0.5, -0.5]) - 1

        offsets = np.cumsum(digits, axis=0)
        return offsets

    def lengths(self, vectors: np.ndarray,
                metric_tensor: np.ndarray) -> np.ndarray:
        """Calculate vector lengths using the metric tensor (Dunitz 1078,
        p227).

        Parameters
        ----------
        vectors : np.ndarray[i, j, k]
            Vectors as in fractional coordinates
        metric_tensor : np.ndarray
            Metric tensor for the lattice

        Returns
        -------
        lengths : np.ndarray
            Vector lengths
        """
        tmp = np.dot(vectors, metric_tensor)
        total_displacement = np.einsum('ij,ji->i', tmp, vectors.T)
        assert total_displacement.shape[0] == vectors.shape[0]
        assert total_displacement.ndim == 1
        return np.sqrt(total_displacement)

    @cached_property
    def displacements(self) -> np.ndarray:
        """Calculate displacements from first set of positions.

        Corrects for elements jumping to the next unit cell.

        Parameters
        ----------
        traj_coords : np.array[i, j, k]
            3-dimensional numpy array with dimensions i: time_steps, j: sites, k: coordinates

        Returns
        -------
        displacements : np.ndarray[i, j]
            Displacements from first set of positions.
        """
        offsets = self.cell_offsets_from_coords(self.trajectory_coords)

        corrected_coords = self.trajectory_coords + offsets

        displacements = []

        first = corrected_coords[self.equilibration_steps]

        for disp in corrected_coords[self.equilibration_steps:]:
            diff_vectors = disp - first
            lengths = self.lengths(diff_vectors,
                                   metric_tensor=self.lattice.metric_tensor)
            displacements.append(lengths)

        displacements = np.array(displacements)

        return displacements.T

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

    @cache
    def diff_displacements(self, *, diffusing_element='Li'):
        idx = np.argwhere([e.name == diffusing_element for e in self.species])
        return self.displacements[idx].squeeze()

    @cached_property
    def attempt_frequency(self):
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

        fs = 1 / self.time_step
        speed = np.diff(self.diff_displacements(), prepend=0)

        freq_mean = self.meanfreq(speed, fs=fs)

        attempt_freq = np.mean(freq_mean)
        attempt_freq_std = np.std(freq_mean)

        print(f'{attempt_freq=:g}')
        print(f'{attempt_freq_std=:g}')

        return speed, attempt_freq, attempt_freq_std

    @cached_property
    def speed(self):
        return self.attempt_frequency[0]

    @cached_property
    def attempt_freq(self):
        return self.attempt_frequency[1]

    @cached_property
    def attempt_freq_std(self):
        return self.attempt_frequency[2]

    @cached_property
    def amplitude(self) -> tuple[np.ndarray, float]:
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

        for i, speed_range in enumerate(self.speed):
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

    @cached_property
    def amplitudes(self):
        return self.amplitude[0]

    @cached_property
    def vibration_amplitude(self):
        return self.amplitude[1]
