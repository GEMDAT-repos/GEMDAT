import numpy as np
from gemdat.trajectory import Trajectory
from pymatgen.core import Species

trajectory = Trajectory(
    coords=np.array([
        [[0.5, .45, .9]],
        [[0, .9, .2]],
        [[0.5, .42, .8]],
        [[0, .1, .25]],
        [[0.5, .59, 0]],
    ]),
    lattice=np.eye(3),
    species=[Species('Li')],
)


def test_cell_offsets():
    assert np.array_equal(
        trajectory._cell_offsets(),
        np.array([
            [[0, 0, 0]],
            [[1, 0, 1]],
            [[0, 0, 0]],
            [[1, 0, 1]],
            [[0, 0, 1]],
        ]))


def test_displacements():
    assert np.allclose(
        trajectory.displacements(),
        np.array([[0., 0.73654599, 0.10440307, 0.70356236, 0.17204651]]))


def test_filter():
    li_trajectory = trajectory.filter(diffusing_element='Li')
    assert np.allclose(
        li_trajectory.displacements(),
        np.array([[0., 0.73654599, 0.10440307, 0.70356236, 0.17204651]]))
