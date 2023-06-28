import numpy as np
from gemdat.calculate import Vibration


def test_meanfreq_single_timestep():
    x = np.sin(np.linspace(0, 1, 6))
    ret = Vibration.meanfreq(x)

    expected = np.array([[0.2303359]])

    np.testing.assert_allclose(ret, expected)


def test_meanfreq():
    x = np.array([
        np.sin(np.linspace(0, 1, 6)),
        np.sin(np.linspace(0, 2, 6)),
        np.sin(np.linspace(0, 3, 6)),
    ])
    ret = Vibration.meanfreq(x)

    expected = np.array([[0.2303359], [0.21308077], [0.17074241]])

    np.testing.assert_allclose(ret, expected)
