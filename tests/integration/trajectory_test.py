from math import isclose

import numpy as np
import pytest
from gemdat.simulation_metrics import SimulationMetrics
from gemdat.volume import trajectory_to_volume


@pytest.vaspxml_available
def test_volume(vasp_traj):
    trajectory = vasp_traj

    diff_trajectory = trajectory.filter('Li')

    vol = trajectory_to_volume(trajectory=diff_trajectory, resolution=0.2)

    assert isinstance(vol, np.ndarray)
    assert vol.shape == (101, 51, 51)
    assert vol.sum() == len(diff_trajectory.species) * len(diff_trajectory)


@pytest.vaspxml_available
def test_volume_cartesian(vasp_traj):
    trajectory = vasp_traj

    diff_trajectory = trajectory.filter('Li')

    vol = trajectory_to_volume(trajectory=diff_trajectory,
                               resolution=0.2,
                               cartesian=True)

    assert isinstance(vol, np.ndarray)
    assert vol.shape == (101, 51, 52)
    assert vol.sum() == len(diff_trajectory.species) * len(diff_trajectory)


@pytest.vaspxml_available
def test_tracer(vasp_traj):

    diff_trajectory = vasp_traj.filter('Li')
    metrics = SimulationMetrics(diff_trajectory)

    assert isclose(metrics.particle_density(), 2.4557e28, rel_tol=1e-4)
    assert isclose(metrics.mol_per_liter(), 40.777, rel_tol=1e-4)
    assert isclose(
        metrics.tracer_diffusivity(diffusion_dimensions=3),
        1.5706e-09,
        rel_tol=1e-4,
    )
    assert isclose(
        metrics.tracer_conductivity(z_ion=1, diffusion_dimensions=3),
        110.322,
        rel_tol=1e-4,
    )