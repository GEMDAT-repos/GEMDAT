from __future__ import annotations

import numpy as np
import pytest

from gemdat.utils import bfill, ffill, meanfreq


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

    np.testing.assert_equal(ret, expected)


def test_bfill(arr):
    ret = bfill(arr)
    expected = np.array([
        [5, 7, 7, 7, 2],
        [3, 1, 1, 8, -1],
        [4, 9, 6, -1, -1],
    ])

    np.testing.assert_equal(ret, expected)


def test_ffill_axis0(arr):
    ret = ffill(arr, axis=0)
    expected = np.array([
        [5, -1, -1, 7, 2],
        [3, -1, 1, 8, 2],
        [4, 9, 6, 8, 2],
    ])

    np.testing.assert_equal(ret, expected)


def test_bfill_axis0(arr):
    ret = bfill(arr, axis=0)
    expected = np.array([
        [5, 9, 1, 7, 2],
        [3, 9, 1, 8, -1],
        [4, 9, 6, -1, -1],
    ])

    np.testing.assert_equal(ret, expected)


def test_meanfreq_single_timestep():
    x = np.sin(np.linspace(0, 1, 6))
    ret = meanfreq(x)

    expected = np.array([[0.2303359]])

    np.testing.assert_allclose(ret, expected)


def test_meanfreq():
    x = np.array([
        np.sin(np.linspace(0, 1, 6)),
        np.sin(np.linspace(0, 2, 6)),
        np.sin(np.linspace(0, 3, 6)),
    ])
    ret = meanfreq(x)

    expected = np.array([[0.2303359], [0.21308077], [0.17074241]])

    np.testing.assert_allclose(ret, expected)
