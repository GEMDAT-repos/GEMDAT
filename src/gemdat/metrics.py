"""This module contains classes for calculating metrics and other properties
from trajectories."""

from __future__ import annotations

import typing

import numpy as np
import uncertainties as u
from pymatgen.core.units import FloatWithUnit
from scipy.constants import Avogadro, Boltzmann, angstrom, elementary_charge

from .caching import weak_lru_cache
from .utils import meanfreq

from ._plot_backend import plot_backend

if typing.TYPE_CHECKING:
    from trajectory import Trajectory


class TrajectoryMetrics:
    """Class for calculating different metrics and properties from a molecular
    dynamics simulation."""

    def __init__(self, trajectory: Trajectory):
        """Initialize class.

        Parameters
        ----------
        trajectory: Trajectory
            Input trajectory
        """
        self.trajectory = trajectory

    @weak_lru_cache()
    def speed(self) -> np.ndarray:
        """Calculate speed.

        Corresponds to change in distance from the base position.

        Returns
        -------
        speed : np.ndarray
            Output array with speeds
        """
        distances = self.trajectory.distances_from_base_position()
        return np.diff(distances, prepend=0)

    @weak_lru_cache()
    def particle_density(self) -> FloatWithUnit:
        """Calculate number of particles per unit of volume from trajectory.

        Returns
        -------
        particle_density : FloatWithUnit
            Number of particles in $m^{-3}$
        """
        lattice = self.trajectory.get_lattice()
        volume_ang = lattice.volume
        volume_m3 = volume_ang * angstrom**3
        particle_density = len(self.trajectory.species) / volume_m3
        return FloatWithUnit(particle_density, 'm^-3')

    @weak_lru_cache()
    def mol_per_liter(self) -> FloatWithUnit:
        """Calculate density.

        Returns
        -------
        particle_density : FloatWithUnit
            Particle density as $mol/l$.
        """
        mol_per_liter = (self.particle_density() * 1e-3) / Avogadro
        return FloatWithUnit(mol_per_liter, 'mol l^-1')

    @weak_lru_cache()
    def tracer_diffusivity(self, *, dimensions: int = 3) -> FloatWithUnit:
        """Calculate tracer diffusivity.

        Defined as: MSD / (2*dimensions*time)

        Parameters
        ----------
        dimensions : int
            Number of diffusion dimensions

        Returns
        -------
        tracer_diffusivity : FloatWithUnit
            Tracer diffusivity in $m^2/s$
        """
        distances = self.trajectory.distances_from_base_position()
        msd = np.mean(distances[:, -1] ** 2)  # Angstrom^2

        tracer_diff = (msd * angstrom**2) / (2 * dimensions * self.trajectory.total_time)

        return FloatWithUnit(tracer_diff, 'm^2 s^-1')

    @weak_lru_cache()
    def tracer_diffusivity_center_of_mass(
        self,
        *,
        dimensions: int = 3,
    ) -> FloatWithUnit:
        """Calculate the tracer diffusivity of the center of mass.

        Parameters
        ----------
        dimensions : int
            Number of diffusion dimensions

        Returns
        -------
        tracer_diffusivity : FloatWithUnit
            Tracer diffusivity in $m^2/s$
        """
        center_of_mass = self.trajectory.center_of_mass()

        metrics = TrajectoryMetrics(center_of_mass)

        return metrics.tracer_diffusivity(dimensions=dimensions)

    @weak_lru_cache()
    def haven_ratio(self, *, dimensions: int = 3) -> float:
        """Calculate Haven's ratio.

        Parameters
        ----------
        dimensions : int
            Number of diffusion dimensions

        Returns
        -------
        haven_ratio : float
        """
        return self.tracer_diffusivity(
            dimensions=dimensions
        ) / self.tracer_diffusivity_center_of_mass(dimensions=dimensions)

    @weak_lru_cache()
    def tracer_conductivity(self, *, z_ion: int, dimensions: int = 3) -> FloatWithUnit:
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
        tracer_conductivity : FloatWithUnit
            Tracer conductivity in $S/m$
        """
        temperature = self.trajectory.metadata['temperature']
        tracer_diff = self.tracer_diffusivity(dimensions=dimensions)
        tracer_conduc = (
            (elementary_charge**2) * (z_ion**2) * tracer_diff * self.particle_density()
        ) / (Boltzmann * temperature)

        return FloatWithUnit(tracer_conduc, 'S m^-1')

    @weak_lru_cache()
    def attempt_frequency(self) -> tuple[FloatWithUnit, FloatWithUnit]:
        """Return attempt frequency and standard deviation in Hz.

        Returns
        -------
        attempt_freq : FloatWithUnit
            Attempt frequency
        attempt_freq_std : FloatWithUnit
            Attempt frequency standard deviation
        """
        speed = self.speed()

        freq_mean = meanfreq(speed, fs=self.trajectory.sampling_frequency)

        attempt_freq_std = np.std(freq_mean)
        attempt_freq_std = FloatWithUnit(attempt_freq_std, 'hz')

        attempt_freq = np.mean(freq_mean)
        attempt_freq = FloatWithUnit(attempt_freq, 'hz')

        return attempt_freq, attempt_freq_std

    @weak_lru_cache()
    def vibration_amplitude(self) -> FloatWithUnit:
        """Calculate vibration amplitude.

        Returns
        -------
        vibration_amp : FloatWithUnit
            Vibration amplitude in $Å$
        """
        amplitudes = self.amplitudes()

        mean_vib = np.mean(amplitudes)
        vibration_amp = np.std(amplitudes)

        mean_vib = FloatWithUnit(mean_vib, 'ang')
        vibration_amp = FloatWithUnit(vibration_amp, 'ang')

        return vibration_amp

    @weak_lru_cache()
    def amplitudes(self) -> np.ndarray:
        """Calculate vibration amplitudes.

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
            subarrays = np.array_split(speed_range, splits[1:-1] + 1)

            amplitudes.extend([np.sum(array) for array in subarrays])

        return np.asarray(amplitudes)


class TrajectoryMetricsStd:
    """Class for calculating different metrics and properties from a molecular
    dynamics simulation.

    Calculates the mean and standard deviation for a given list of
    trajectories
    """

    def __init__(self, trajectories: list[Trajectory]):
        """Initialize class.

        Parameters
        ----------
        trajectories: list[Trajectory]
            Input trajectories
        """
        self.metrics = [TrajectoryMetrics(trajectory) for trajectory in trajectories]

    def speed(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculate mean speed and standard deviations.

        Corresponds to change in distance from the base position.

        Returns
        -------
        speed_mean, speed_std : tuple[np.ndarray, np.ndarray]
            Output arrays with speeds
        """
        speeds = [metric.speed() for metric in self.metrics]
        return (np.mean(speeds, axis=0), np.std(speeds, axis=0))

    def tracer_diffusivity(self, *, dimensions: int) -> u.ufloat:
        """Calculate tracer diffusivity.

        Defined as: MSD / (2*dimensions*time)

        Parameters
        ----------
        dimensions : int
            Number of diffusion dimensions

        Returns
        -------
        tracer_diffusivity : u.ufloat
            Tracer diffusivity in $m^2/s$, mean and standard deviation
        """
        diffusivities = [
            metric.tracer_diffusivity(dimensions=dimensions) for metric in self.metrics
        ]
        mean_diffusivities = FloatWithUnit(np.mean(diffusivities), 'm^2 s^-1')
        std_diffusivities = FloatWithUnit(np.std(diffusivities), 'm^2 s^-1')
        return u.ufloat(mean_diffusivities, std_diffusivities)

    def tracer_conductivity(self, *, z_ion: int, dimensions: int) -> u.ufloat:
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
        tracer_conductivities : u.ufloat
            Tracer conductivities in $S/m$, mean and standard deviation
        """
        conductivities = [
            metric.tracer_conductivity(z_ion=z_ion, dimensions=dimensions)
            for metric in self.metrics
        ]
        mean_conductivities = FloatWithUnit(np.mean(conductivities), 'S m^-1')
        std_conductivities = FloatWithUnit(np.std(conductivities), 'S m^-1')
        return u.ufloat(mean_conductivities, std_conductivities)

    def vibration_amplitude(self) -> u.ufloat:
        """Calculate vibration amplitude.

        Returns
        -------
        vibration_amp_mean, vibration_amp_std : tuple[FloatWithUnit, FloatWithUnit]
            Vibration amplitude in $Å$, mean and standard deviation
        """
        vibes = [metric.vibration_amplitude() for metric in self.metrics]
        mean_vibes = FloatWithUnit(np.mean(vibes), 'ang')
        standard_vibes = FloatWithUnit(np.std(vibes), 'ang')  # Standard deviation
        return u.ufloat(mean_vibes, standard_vibes)

    def amplitudes(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculate vibration amplitudes.

        Returns
        -------
        amplitudes_mean, amplitudes_std : tuple[np.ndarray, np.ndarray]
            Output array of vibration amplitudes, mean and standard deviation
        """
        amplitudes = [metric.amplitudes() for metric in self.metrics]
        return (np.mean(amplitudes, axis=0), np.std(amplitudes, axis=0))


class ArrheniusFit:
    """Arrhenius fit result for diffusion coefficients.

    The model is:
        D(T) = D0 * exp(-Ea / (k_B * T))

    where the fit is performed in log-space:
        ln D = ln D0 + m * (1/T)

    with:
        m = -Ea / k_B

    Notes
    -----
    If `diffusivities_std` is provided, the fit is weighted using an
    approximate log-uncertainty:

        sigma_lnD ≈ sigma_D / D
    """

    def __init__(
        self,
        *,
        temperatures,
        diffusivities,
        diffusivities_std=None,
        particle_density=None,
        diffusivity_unit: str = "m^2 s^-1",
    ):
        self.temperatures = np.asarray(temperatures, dtype=float)
        self.diffusivities = np.asarray(diffusivities, dtype=float)
        self.diffusivities_std = (
            None if diffusivities_std is None else np.asarray(diffusivities_std, dtype=float)
        )

        if self.temperatures.shape[0] != self.diffusivities.shape[0]:
            raise ValueError("Temperatures and diffusivities must have the same length")
        if self.temperatures.size < 2:
            raise ValueError("Need at least two points for Arrhenius fit.")
        if np.any(self.diffusivities <= 0):
            raise ValueError("Diffusivities must be > 0 for log-space fit.")
        if diffusivities_std is not None and self.diffusivities_std.shape != self.diffusivities.shape:
                raise ValueError("diffusivities_std must have the same shape as diffusivities.")

        self.particle_density = particle_density
        self.diffusivity_unit = diffusivity_unit

        self.slope = 0.0
        self.intercept = 0.0
        self.cov = np.zeros((2, 2), dtype=float)

        self._fit()

    @classmethod
    def from_trajectories(
        cls,
        trajectories,
        *,
        diffusing_specie: str,
        dimensions: int = 3,
        n_parts: int = 10,
        equal_parts: bool = True,
        diffusivity_unit: str = "m^2 s^-1",
    ) -> "ArrheniusFit":
        """Fit Arrhenius parameters from trajectories at different temperatures.

        Parameters
        ----------
        trajectories
            List of trajectories, one per temperature. Each trajectory must have
            `trajectory.metadata['temperature']` in K.
        diffusing_specie
            Element symbol of diffusing species (e.g. "Li").
        dimensions
            Number of diffusion dimensions used for tracer diffusivity.
        n_parts
            Number of parts to split each diffusing trajectory into for estimating
            mean and standard deviation.
        equal_parts
            If True, split into equal-length parts.
        diffusivity_unit
            Unit label stored on the fit output.

        Returns
        -------
        fit
            ArrheniusFit instance.
        """
        temperatures = []
        diffusivities = []
        diffusivities_std = []

        # Use the first trajectory to store particle density (m^-3) of the diffusing species
        diff_traj_0 = trajectories[0].filter(diffusing_specie)
        particle_density = TrajectoryMetrics(diff_traj_0).particle_density()

        for traj in trajectories:
            temperature = float(traj.metadata["temperature"])
            diff_traj = traj.filter(diffusing_specie)

            parts = diff_traj.split(n_parts=n_parts, equal_parts=equal_parts)
            metrics_std = TrajectoryMetricsStd(trajectories=parts)
            d = metrics_std.tracer_diffusivity(dimensions=dimensions)

            temperatures.append(temperature)
            diffusivities.append(float(d.n))
            diffusivities_std.append(float(d.s))

        order = np.argsort(temperatures)
        temperatures = np.asarray(temperatures, dtype=float)[order]
        diffusivities = np.asarray(diffusivities, dtype=float)[order]
        diffusivities_std = np.asarray(diffusivities_std, dtype=float)[order]

        return cls(
            temperatures=temperatures,
            diffusivities=diffusivities,
            diffusivities_std=diffusivities_std,
            particle_density=particle_density,
            diffusivity_unit=diffusivity_unit,
        )

    def _fit(self) -> None:
        x = 1.0 / self.temperatures
        y = np.log(self.diffusivities)

        a = np.column_stack((x, np.ones_like(x)))

        if self.diffusivities_std is None:
            w = np.ones_like(y)
        else:
            sigma_lnD = self.diffusivities_std / self.diffusivities
            w = 1.0 / (sigma_lnD * sigma_lnD)

        sw = np.sqrt(w)
        aw = a * sw[:, None]
        yw = y * sw

        beta, _, _, _ = np.linalg.lstsq(aw, yw, rcond=None)
        self.slope = float(beta[0])
        self.intercept = float(beta[1])

        resid = y - (a @ beta)
        rss = float(np.sum(w * resid * resid))
        n = y.shape[0]

        atwa = a.T @ (a * w[:, None])
        # scale by residual variance
        self.cov = np.linalg.inv(atwa) * (rss / (n - 2))

    def activation_energy(self) -> u.ufloat:
        """Return activation energy (eV) as ufloat(mean, std)."""
        ea = -self.slope * Boltzmann / elementary_charge
        ea_std = np.sqrt(self.cov[0, 0]) * Boltzmann / elementary_charge
        return u.ufloat(FloatWithUnit(ea, "eV"), FloatWithUnit(ea_std, "eV"))

    def prefactor(self) -> u.ufloat:
        """Return prefactor D0 as ufloat(mean, std) in `diffusivity_unit`."""
        d0 = float(np.exp(self.intercept))
        d0_std = d0 * float(np.sqrt(self.cov[1, 1]))
        return u.ufloat(
            FloatWithUnit(d0, self.diffusivity_unit),
            FloatWithUnit(d0_std, self.diffusivity_unit),
        )

    def extrapolate_diffusivity(self, temperature: float) -> u.ufloat:
        """Extrapolate diffusivity at given temperature (K), with uncertainty."""
        x = 1.0 / float(temperature)
        v = np.array([x, 1.0], dtype=float)

        ln_d = self.intercept + self.slope * x
        var_ln_d = float(v @ self.cov @ v)
        std_ln_d = float(np.sqrt(var_ln_d))

        d = float(np.exp(ln_d))
        d_std = d * std_ln_d

        return u.ufloat(
            FloatWithUnit(d, self.diffusivity_unit),
            FloatWithUnit(d_std, self.diffusivity_unit),
        )

    def extrapolate_conductivity(self, temperature: float, *, z_ion: int) -> u.ufloat:
        """Extrapolate tracer conductivity at given temperature (K), with uncertainty."""
        if self.particle_density is None:
            raise ValueError("particle_density is not set on this ArrheniusFit")

        d = self.extrapolate_diffusivity(float(temperature))
        factor = (elementary_charge**2) * (z_ion**2) * self.particle_density / (
            Boltzmann * float(temperature)
        )

        sigma = float(factor * d.n)
        sigma_std = float(factor * d.s)

        return u.ufloat(FloatWithUnit(sigma, "S m^-1"), FloatWithUnit(sigma_std, "S m^-1"))

    @plot_backend
    def plot_arrhenius(self, *, module, **kwargs):
        """See [gemdat.plots.arrhenius][] for more info."""
        return module.arrhenius(fit=self, **kwargs)

