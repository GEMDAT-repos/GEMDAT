import numpy as np
from gemdat.sites import _calculate_transitions_matrix


def test_transitions_matrix():
    a = np.array([
        [0, 0, 1, 10, 20],
        [0, 1, 0, 20, 30],
        [1, 2, 1, 5, 6],
        [1, 1, 0, 6, 7],
        [2, 2, 0, 30, 50],
    ])

    n = 3
    transitions = _calculate_transitions_matrix(a, n_diffusing=n)

    assert transitions.shape == (n, n)
    assert transitions.dtype == int
    np.testing.assert_equal(transitions,
                            np.array([[0, 1, 0], [2, 0, 0], [1, 1, 0]]))
