from types import SimpleNamespace

import numpy as np

from .calculate.displacements import Displacements
from .calculate.tracer import Tracer
from .calculate.vibration import Vibration
from .trajectory import GemdatTrajectory


def calculate_all(trajectory: GemdatTrajectory,
                  *,
                  diffusing_element: str,
                  known_structure: str | None = None,
                  equilibration_steps: int = 1250,
                  diffusion_dimensions: int = 3,
                  z_ion: float = 1.0,
                  n_parts: int = 10,
                  dist_collective: float = 4.5):
    """Calculate extra parameters and return them as a simple namespace.

    Parameters
    ----------
    trajectory : GemdatTrajectory
        Input trajectory coordinates
    diffusing_element : str
        Name of the diffusing element
    structure : str | None
        Path to cif file or name of known structure
    equilibration_steps : int
        Number of equilibration steps
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
        equilibration_steps=equilibration_steps,
        diffusion_dimensions=diffusion_dimensions,
        z_ion=z_ion,
        n_parts=n_parts,
        dist_collective=dist_collective,
    )
    trajectory = trajectory[equilibration_steps:]

    _add_shared_variables(trajectory, extras)

    extras.__dict__.update(
        Displacements.calculate_all(trajectory, extras=extras))
    extras.__dict__.update(Vibration.calculate_all(trajectory, extras=extras))
    extras.__dict__.update(Tracer.calculate_all(trajectory, extras=extras))

    return extras


def _add_shared_variables(trajectory: GemdatTrajectory,
                          extras: SimpleNamespace):
    """Add common shared variables to extras namespace."""
    extras.n_diffusing = sum(
        [e.name == extras.diffusing_element for e in trajectory.species])
    extras.n_steps = len(trajectory)

    extras.total_time = extras.n_steps * trajectory.time_step

    diffusing_idx = np.argwhere([
        e.name == extras.diffusing_element for e in trajectory.species
    ]).flatten()

    extras.diff_coords = trajectory.coords[:, diffusing_idx, :]
