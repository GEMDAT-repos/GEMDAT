from types import SimpleNamespace

import numpy as np
from gemdat import SimulationData
from gemdat.calculate import Vibration
from gemdat.calculate.vibration import meanfreq

data = SimulationData(
    time_step=1,
    trajectory_coords=None,
    lattice=None,
    species=None,
    temperature=None,
    parameters=None,
    structure=None,
)
extras = SimpleNamespace(diff_displacements=np.array(
    [[0., 0.73654599, 0.10440307, 0.70356236, 0.17204651]]), )


def test_vibration_calculate_all():
    ret = Vibration.calculate_all(data, extras)
    assert (np.allclose(
        ret['speed'],
        np.array([[0., 0.73654599, -0.63214292, 0.59915929, -0.53151585]])))
    assert (np.isclose(ret['attempt_freq'], 0.38779581521997086))
    assert (np.isclose(ret['attempt_freq_std'], 0.))
    assert (np.allclose(
        ret['amplitudes'],
        np.array([0.73654599, -0.63214292, 0.59915929, -0.53151585])))
    assert (np.isclose(ret['vibration_amplitude'], 0.6277351391878859))


def test_meanfreq_single_timestep():
    x = np.sin(np.linspace(0, 1, 6))
    ret = meanfreq(x)

    expected = np.array([[0.2303359]])

    np.testing.assert_allclose(ret, expected)


def test_meanfreq():
    x = np.array([
        np.sin(np.linspace(0, 1, 6)),
        np.sin(np.linspace(0, 2, 6)),
        np.sin(np.linspace(0, 3, 6)),
    ])
    ret = meanfreq(x)

    expected = np.array([[0.2303359], [0.21308077], [0.17074241]])

    np.testing.assert_allclose(ret, expected)
