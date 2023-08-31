import numpy as np
from gemdat.segmentation import watershed_pbc


def test_watershed06():
    data = np.array(
        [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 1, 1, 0, 0],
         [0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 1, 1, 0, 0],
         [0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]],
        dtype=int)

    mask = data == 0

    markers = np.zeros_like(data, dtype=int)
    markers[4, 0] = 1

    out = watershed_pbc(data, markers, mask=mask)

    expected = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1],
                         [1, 1, 1, 0, 0, 1, 1], [1, 1, 1, 0, 0, 1, 1],
                         [1, 1, 1, 0, 0, 1, 1], [1, 1, 1, 0, 0, 1, 1],
                         [1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0]])

    np.testing.assert_allclose(out, expected)
