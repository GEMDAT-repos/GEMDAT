from __future__ import annotations

import typing
from types import SimpleNamespace

from .tracer import Tracer
from .vibration import Vibration

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
    )
    extras.n_steps = len(trajectory)
    extras.total_time = extras.n_steps * trajectory.time_step

    _ = trajectory.to_displacements()

    diff_trajectory = trajectory.filter(extras.diffusing_element)

    extras.diff_displacements = diff_trajectory.total_distances()
    extras.n_diffusing = len(diff_trajectory.species)

    extras.__dict__.update(Vibration.calculate_all(diff_trajectory))
    extras.__dict__.update(Tracer.calculate_all(trajectory, extras=extras))

    return extras
