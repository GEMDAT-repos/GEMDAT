import numpy as np
from gemdat.trajectory import Trajectory

trajectory = Trajectory(
    coords=np.array([
        [[0., .0, .1]],
        [[0., .0, .2]],
        [[0., .0, .4]],
        [[0., .0, .5]],
        [[0., .0, .6]],
        [[0., .0, .7]],
    ]),
    species=['Li'],
    time_step=1,
    lattice=np.eye(3) * 10e10,
)
trajectory.temperature = 1


def test_vibration_calculate_all():
    assert (np.isclose(trajectory.particle_density(), 0.001))
    assert (np.isclose(trajectory.mol_per_liter, 1.6605390671738466e-30))
    assert (np.isclose(trajectory.tracer_diffusivity(), 1.))
    assert (np.isclose(trajectory.tracer_conductivity(),
                       9.172294469819148e-41))
