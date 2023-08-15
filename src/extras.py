from __future__ import annotations

import typing
from types import SimpleNamespace

from .simulation_metrics import SimulationMetrics

if typing.TYPE_CHECKING:
    from gemdat.trajectory import Trajectory


def calculate_all(trajectory: Trajectory,
                  *,
                  diffusing_element: str,
                  known_structure: str | None = None,
                  diffusion_dimensions: int = 3,
                  z_ion: float = 1.0,
                  n_parts: int = 10,
                  dist_collective: float = 4.5):
    """Calculate extra parameters and return them as a simple namespace.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory coordinates
    diffusing_element : str
        Name of the diffusing element
    structure : str | None
        Path to cif file or name of known structure
    diffusion_dimensions : int
        Number of diffusion dimensions
    z_ion : float
        Ionic charge of the diffusing ion
    n_parts : int
        In how many parts to divide your simulation for statistics
    dist_collective : float
        Maximum distance for collective motions in Angstrom
    """
    extras = SimpleNamespace(
        diffusing_element=diffusing_element,
        known_structure=known_structure,
        diffusion_dimensions=diffusion_dimensions,
        z_ion=z_ion,
        n_parts=n_parts,
        dist_collective=dist_collective,
        n_steps=len(trajectory),
        total_time=trajectory.total_time,
    )

    diff_trajectory = trajectory.filter(diffusing_element)
    metrics = SimulationMetrics(diff_trajectory)

    attempt_freq, attempt_freq_std = metrics.attempt_frequency()

    extras.attempt_freq = attempt_freq
    extras.attempt_freq_std = attempt_freq_std
    extras.amplitudes = metrics.amplitudes()
    extras.vibration_amplitude = metrics.vibration_amplitude()
    extras.particle_density = metrics.particle_density()
    extras.mol_per_liter = metrics.mol_per_liter()
    extras.tracer_diff = metrics.tracer_diffusivity(diffusion_dimensions=3)
    extras.tracer_conduc = metrics.tracer_conductivity(z_ion=1,
                                                       diffusion_dimensions=3)

    return extras
