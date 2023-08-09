from types import SimpleNamespace

import numpy as np
from gemdat.calculate import Displacements


def test_displacement_calculate_all(trajectory):
    extras = SimpleNamespace(diffusing_element='B')

    ret = Displacements.calculate_all(trajectory, extras)

    assert np.array_equal(
        ret['cell_offsets'],
        np.array([[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]],
                  [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]],
                  [[1, 0, 0], [0, 0, 0]]]))

    assert np.allclose(
        ret['displacements'],
        np.array([[0., 0.2, 0.4, 0.6, 0.9], [0., 0., 0., 0., 0.]]))

    assert np.allclose(ret['diff_displacements'],
                       np.array([0., 0.2, 0.4, 0.6, 0.9]))
