from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from gemdat.shape import ShapeData


@pytest.fixture
def shape():
    name = 'test'
    coords = np.array([
        [0, 3, 4],
        [5, 0, 12],
        [8, 15, 0],
    ])
    return ShapeData(name=name, coords=coords, radius=20)


def test_distances(shape):
    dists = shape.distances()
    assert_allclose(dists, [5, 13, 17])


def test_xyz(shape):
    assert_allclose(shape.x, [0, 5, 8])
    assert_allclose(shape.y, [3, 0, 15])
    assert_allclose(shape.z, [4, 12, 0])


def test_centroid(shape):
    assert_allclose(shape.centroid(), 5.742857, 7.714286, 5.028571)
