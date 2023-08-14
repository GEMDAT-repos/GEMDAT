import numpy as np


def test_displacement_calculate_all(trajectory):
    assert np.allclose(
        trajectory.distances_from_base_position(),
        np.array([[0., 0.2, 0.4, 0.6, 0.9], [0., 0., 0., 0., 0.]]))

    diff_trajectory = trajectory.filter('B')

    assert np.allclose(diff_trajectory.distances_from_base_position(),
                       np.array([0., 0.2, 0.4, 0.6, 0.9]))
