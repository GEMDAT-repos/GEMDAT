from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from gemdat.utils import bfill, ffill, integer_remap, meanfreq


@pytest.fixture
def arr():
    return np.array([
        [5, -1, -1, 7, 2],
        [3, -1, 1, 8, -1],
        [4, 9, 6, -1, -1],
    ])


def test_ffill(arr):
    ret = ffill(arr)
    expected = np.array([
        [5, 5, 5, 7, 2],
        [3, 3, 1, 8, 8],
        [4, 9, 6, 6, 6],
    ])

    assert_equal(ret, expected)


def test_bfill(arr):
    ret = bfill(arr)
    expected = np.array([
        [5, 7, 7, 7, 2],
        [3, 1, 1, 8, -1],
        [4, 9, 6, -1, -1],
    ])

    assert_equal(ret, expected)


def test_ffill_axis0(arr):
    ret = ffill(arr, axis=0)
    expected = np.array([
        [5, -1, -1, 7, 2],
        [3, -1, 1, 8, 2],
        [4, 9, 6, 8, 2],
    ])

    assert_equal(ret, expected)


def test_bfill_axis0(arr):
    ret = bfill(arr, axis=0)
    expected = np.array([
        [5, 9, 1, 7, 2],
        [3, 9, 1, 8, -1],
        [4, 9, 6, -1, -1],
    ])

    assert_equal(ret, expected)


def test_integer_remap():
    a = np.array([4, 2, 1, 3])
    key = np.array([10, 20, 30, 40])
    ret = integer_remap(a, key=key)
    assert_equal(ret, a * 10)


def test_meanfreq_single_timestep():
    x = np.sin(np.linspace(0, 1, 6))
    ret = meanfreq(x)

    expected = np.array([[0.2303359]])

    assert_allclose(ret, expected)


def test_meanfreq():
    x = np.array([
        np.sin(np.linspace(0, 1, 6)),
        np.sin(np.linspace(0, 2, 6)),
        np.sin(np.linspace(0, 3, 6)),
    ])
    ret = meanfreq(x)

    expected = np.array([[0.2303359], [0.21308077], [0.17074241]])

    assert_allclose(ret, expected)
