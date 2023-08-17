from __future__ import annotations

import typing
from functools import lru_cache

import numpy as np
from pymatgen.core.units import FloatWithUnit
from scipy.constants import Avogadro, Boltzmann, angstrom, elementary_charge

from .utils import meanfreq

if typing.TYPE_CHECKING:
    from gemdat.trajectory import Trajectory


class SimulationMetrics:

    def __init__(self, trajectory: Trajectory):
        self.trajectory = trajectory

    @lru_cache
    def speed(self) -> np.ndarray:
        """Calculate speed.

        Corresponds to change in distance from the base position.
        """
        distances = self.trajectory.distances_from_base_position()
        return np.diff(distances, prepend=0)

    @lru_cache
    def particle_density(self) -> float:
        """Return number of particles per m3."""
        lattice = self.trajectory.get_lattice()
        volume_ang = lattice.volume
        volume_m3 = volume_ang * angstrom**3
        particle_density = len(self.trajectory.species) / volume_m3
        return FloatWithUnit(particle_density, 'm^-3')

    @lru_cache
    def mol_per_liter(self) -> float:
        """Return particle density as mol/liter."""
        mol_per_liter = (self.particle_density() * 1e-3) / Avogadro
        return FloatWithUnit(mol_per_liter, 'mol l^-1')

    @lru_cache
    def tracer_diffusivity(self, *, dimensions: int) -> float:
        """Return tracer diffusivity in m2/s.

        Defined as: MSD/(2*dimensions*time)

        Parameters
        ----------
        dimensions : int
            Number of diffusion dimensions

        Returns
        -------
        tracer_diffusivity : float
        """
        distances = self.trajectory.distances_from_base_position()
        msd = np.mean(distances[:, -1]**2)  # Angstrom^2

        tracer_diff = (msd * angstrom**2) / (2 * dimensions *
                                             self.trajectory.total_time)

        return FloatWithUnit(tracer_diff, 'm^2 s^-1')

    @lru_cache
    def tracer_conductivity(self, *, z_ion: int, dimensions: int) -> float:
        """Return tracer conductivity as S/m.

        Defined as: elementary_charge^2 * charge_ion^2 * diffusivity *
            particle_density / (k_B * T)

        Parameters
        ----------
        z_ion : int
            Charge of the ion
        dimensions : int
            Number of diffusion dimensions

        Returns
        -------
        tracer_conductivity : float
        """
        temperature = self.trajectory.metadata['temperature']
        tracer_diff = self.tracer_diffusivity(dimensions=dimensions)
        tracer_conduc = ((elementary_charge**2) * (z_ion**2) * tracer_diff *
                         self.particle_density()) / (Boltzmann * temperature)

        return FloatWithUnit(tracer_conduc, 'S m^-1')

    @lru_cache
    def attempt_frequency(self) -> tuple[float, float]:
        """Return attempt frequency and standard deviation in Hz.

        Returns
        -------
        attempt_freq : float
            Attempt frequency
        attempt_freq_std : float
            Attempt frequency standard deviation
        """
        speed = self.speed()

        freq_mean = meanfreq(speed, fs=self.trajectory.sampling_frequency)

        attempt_freq_std = np.std(freq_mean)
        attempt_freq_std = FloatWithUnit(attempt_freq_std, 'hz')

        attempt_freq = np.mean(freq_mean)
        attempt_freq = FloatWithUnit(attempt_freq, 'hz')

        return attempt_freq, attempt_freq_std

    @lru_cache
    def vibration_amplitude(self) -> float:
        """Return vibration amplitude in Angstrom.

        Returns
        -------
        vibration_amp : float
        """
        amplitudes = self.amplitudes()

        mean_vib = np.mean(amplitudes)
        vibration_amp = np.std(amplitudes)

        mean_vib = FloatWithUnit(mean_vib, 'ang')
        vibration_amp = FloatWithUnit(vibration_amp, 'ang')

        return vibration_amp

    @lru_cache
    def amplitudes(self) -> np.ndarray:
        """Return vibration amplitudes.

        Returns
        -------
        amplitudes : np.ndarray
            Output array of vibration amplitudes
        """
        amplitudes = []
        speed = self.speed()

        for i, speed_range in enumerate(speed):
            signs = np.sign(speed_range)

            # get indices where sign flips
            splits = np.where(signs != np.roll(signs, shift=-1))[0]
            # strip first and last splits
            subarrays = np.split(speed_range, splits[1:-1] + 1)

            amplitudes.extend([np.sum(array) for array in subarrays])

        return np.asarray(amplitudes)
