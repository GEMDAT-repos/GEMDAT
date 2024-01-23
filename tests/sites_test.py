from __future__ import annotations

import numpy as np
import pandas as pd

from gemdat.transitions import _calculate_transitions_matrix


def test_transitions_matrix():
    a = np.array([
        [0, 0, 1, 10],
        [0, 1, 0, 20],
        [1, 2, 1, 5],
        [1, 1, 0, 6],
        [2, 2, 0, 30],
    ])
    a = pd.DataFrame(
        data=a,
        columns=['atom index', 'start site', 'destination site', 'time'])

    n = 3
    transitions = _calculate_transitions_matrix(a, n_sites=n)

    assert transitions.shape == (n, n)
    assert transitions.dtype == int
    np.testing.assert_equal(transitions,
                            np.array([[0, 1, 0], [2, 0, 0], [1, 1, 0]]))
