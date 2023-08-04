import numpy as np
from gemdat import SitesData, Trajectory, Vibration
from pymatgen.core import Species

trajectory = Trajectory(coords=np.array([
    [[0., .0, .1], [0., .0, .4]],
    [[0., .0, .1], [0., .0, .4]],
    [[0., .0, .1], [0., .0, .4]],
    [[0., .0, .1], [0., .0, .4]],
    [[0., .0, .4], [0., .0, .1]],
    [[0., .0, .4], [0., .0, .1]],
    [[0., .0, .4], [0., .0, .1]],
    [[0., .0, .4], [0., .0, .1]],
]),
                        lattice=np.eye(3),
                        species=[Species('Li')] * 2,
                        time_step=1.)
vibration = Vibration(trajectory, fs=1.0)
sites = SitesData(trajectory.get_structure(0), trajectory, vibration)


def test_transitions_matrix():
    np.array([
        [0, 0, 1, 10, 20],
        [0, 1, 0, 20, 30],
        [1, 2, 1, 5, 6],
        [1, 1, 0, 6, 7],
        [2, 2, 0, 30, 50],
    ])

    n = 2
    transitions = sites.transitions()

    assert transitions.shape == (n, n)
    assert transitions.dtype == int
    np.testing.assert_equal(transitions, np.array([[0, 1], [1, 0]]))
