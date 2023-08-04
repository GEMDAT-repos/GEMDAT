import numpy as np
from gemdat import Trajectory, Vibration
from pymatgen.core import Species

trajectory = Trajectory(
    coords=np.array([
        [[0., .0, .1]],
        [[0., .0, .2]],
        [[0., .0, .1]],
        [[0., .0, .175]],
        [[0., .0, .125]],
        [[0., .0, .175]],
    ]),
    lattice=np.eye(3),
    species=[Species('Li')],
    time_step=1,
)


def test_vibration_calculate_all():
    ret = Vibration(trajectory, fs=1.0)
    assert (np.allclose(ret.speed(),
                        np.array([[0., 0.1, -0.1, 0.075, -0.05, 0.05]])))
    assert (np.isclose(ret.attempt_frequency()[0], 0.4543859649122808))  # freq
    assert (np.isclose(ret.attempt_frequency()[1], 0.))  # std deviation
    assert (np.allclose(ret.amplitudes(),
                        np.array([0.1, -0.1, 0.075, -0.05, 0.05])))
    assert (np.isclose(ret.vibration_amplitude(), 0.07681145747868609))


def test_meanfreq_single_timestep():
    x = np.sin(np.linspace(0, 1, 6))
    ret = Vibration.meanfreq(x, fs=1.0)

    expected = np.array([[0.2303359]])

    np.testing.assert_allclose(ret, expected)


def test_meanfreq():
    x = np.array([
        np.sin(np.linspace(0, 1, 6)),
        np.sin(np.linspace(0, 2, 6)),
        np.sin(np.linspace(0, 3, 6)),
    ])
    ret = Vibration.meanfreq(x, fs=1.0)

    expected = np.array([[0.2303359], [0.21308077], [0.17074241]])

    np.testing.assert_allclose(ret, expected)
