import numpy as np
from gemdat.calculate import Vibration
from gemdat.calculate.vibration import meanfreq


def test_vibration_calculate_all(trajectory):
    ret = Vibration.calculate_all(trajectory.filter('B'))

    assert np.allclose(ret['speed'], np.array([[0., 0.2, 0.2, 0.2, 0.3]]))
    assert np.isclose(ret['attempt_freq'], 0.33727)
    assert np.isclose(ret['attempt_freq_std'], 0.0)
    assert np.allclose(ret['amplitudes'], np.array([0.9]))
    assert np.isclose(ret['vibration_amplitude'], 0.0)


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
