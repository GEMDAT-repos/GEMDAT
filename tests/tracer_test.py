from types import SimpleNamespace

import numpy as np
from gemdat.calculate.tracer import Tracer
from gemdat.trajectory import Trajectory

trajectory = Trajectory(
    coords=[[0, 0, 0], [0, 0, 0]],
    species=['Li'],
    time_step=1,
    lattice=np.eye(3) * 10e10,
)
trajectory.temperature = 1
extras = SimpleNamespace(
    diff_displacements=np.array(
        [[0., 0.73654599, 0.10440307, 0.70356236, 0.17204651]]),
    total_time=1,
    n_diffusing=1,
    diffusion_dimensions=3,
    z_ion=1,
)


def test_vibration_calculate_all():
    ret = Tracer.calculate_all(trajectory, extras)
    assert (np.isclose(ret['particle_density'], 0.001))
    assert (np.isclose(ret['mol_per_liter'], 1.6605390671738466e-30))
    assert (np.isclose(ret['tracer_diff'], 4.933333600530017e-23))
    assert (np.isclose(ret['tracer_conduc'], 9.172294469819148e-41))
