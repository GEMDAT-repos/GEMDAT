from __future__ import annotations

import typing

import numpy as np
from pymatgen.core.units import FloatWithUnit
from scipy.constants import Avogadro, Boltzmann, angstrom, elementary_charge

from .utils import meanfreq

if typing.TYPE_CHECKING:
    from gemdat.trajectory import Trajectory


class SimulationMetrics:

    def __init__(self, trajectory: Trajectory):
        self.trajectory = trajectory

    @property
    def total_time(self):
        return len(self.trajectory) * self.trajectory.time_step

    def particle_density(self):
        lattice = self.trajectory.get_lattice()
        volume_ang = lattice.volume
        volume_m3 = volume_ang * angstrom**3
        particle_density = len(self.trajectory.species) / volume_m3
        return FloatWithUnit(particle_density, 'm^-3')

    def mol_per_liter(self):
        mol_per_liter = (self.particle_density() * 1e-3) / Avogadro
        return FloatWithUnit(mol_per_liter, 'mol l^-1')

    def tracer_diffusivity(self, *, diffusion_dimensions: int):
        # Matlab code contains a bug here so I'm not entirely sure what is the definition
        # Matlab code takes the first column, which is equal to 0
        # Do they mean the total displacement (i.e. last column)?
        msd = np.mean(self.trajectory.distances_from_base_position()[:, -1]**
                      2)  # Angstrom^2

        # Diffusivity = MSD/(2*dimensions*time)
        tracer_diff = (msd * angstrom**2) / (2 * diffusion_dimensions *
                                             self.total_time)

        return FloatWithUnit(tracer_diff, 'm^2 s^-1')

    def tracer_conductivity(self, *, z_ion: int, diffusion_dimensions: int):
        # Conductivity = elementary_charge^2 * charge_ion^2 * diffusivity * particle_density / (k_B * T)
        temperature = self.trajectory.metadata['temperature']
        tracer_diff = self.tracer_diffusivity(
            diffusion_dimensions=diffusion_dimensions)
        tracer_conduc = ((elementary_charge**2) * (z_ion**2) * tracer_diff *
                         self.particle_density()) / (Boltzmann * temperature)

        return FloatWithUnit(tracer_conduc, 'S m^-1')

    @property
    def fs(self) -> float:
        """Return sampling frequency of simulation."""
        return 1 / self.trajectory.time_step

    def attempt_frequency(self) -> tuple[float, float]:
        """Calculate attempt frequency.

        Returns
        -------
        attempt_freq : float
            Attempt frequency
        attempt_freq_std : float
            Attempt frequency standard deviation
        """
        distances = self.trajectory.distances_from_base_position()
        speed = np.diff(distances, prepend=0)

        freq_mean = meanfreq(speed, fs=self.fs)

        attempt_freq_std = np.std(freq_mean)
        attempt_freq_std = FloatWithUnit(attempt_freq_std, 'hz')

        attempt_freq = np.mean(freq_mean)
        attempt_freq = FloatWithUnit(attempt_freq, 'hz')

        return attempt_freq, attempt_freq_std

    def vibration_amplitude(self) -> float:
        """Calculate vibration amplitude.

        Parameters
        ----------
        amplitudes : np.ndarray
            Input amplitudes

        Returns
        -------
        vibration_amp : float
            Vibration amplitude
        """
        amplitudes = self.amplitudes()

        mean_vib = np.mean(amplitudes)
        vibration_amp = np.std(amplitudes)

        mean_vib = FloatWithUnit(mean_vib, 'ang')
        vibration_amp = FloatWithUnit(vibration_amp, 'ang')

        return vibration_amp

    def amplitudes(self) -> np.ndarray:
        """Calculate vibration amplitude.

        Returns
        -------
        amplitudes : np.ndarray
            Output array with vibrational amplitudes
        """
        amplitudes = []

        distances = self.trajectory.distances_from_base_position()
        speed = np.diff(distances, prepend=0)

        for i, speed_range in enumerate(speed):
            signs = np.sign(speed_range)

            # get indices where sign flips
            splits = np.where(signs != np.roll(signs, shift=-1))[0]
            # strip first and last splits
            subarrays = np.split(speed_range, splits[1:-1] + 1)

            amplitudes.extend([np.sum(array) for array in subarrays])

        return np.asarray(amplitudes)
