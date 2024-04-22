from __future__ import annotations

from math import isclose

import numpy as np
import pytest

from gemdat.orientations import calculate_spherical_areas

matrix = np.array(
    [[1 / 2**0.5, -1 / 6**0.5, 1 / 3**0.5],
     [1 / 2**0.5, 1 / 6**0.5, -1 / 3**0.5], [0, 2 / 6**0.5, 1 / 3**0.5]], )


@pytest.vasporicache_available  # type: ignore
def test_normalize(vasp_orientations):
    ret = vasp_orientations.normalize()
    assert isclose(ret.vectors.std(), 0.5773501929401034)
    first = ret.vectors[0, 0]
    np.testing.assert_allclose(first, [0.300804, 0.433833, -0.849297],
                               atol=1e-06)


@pytest.vasporicache_available  # type: ignore
def test_conventional(vasp_orientations):
    ret = vasp_orientations.transform(matrix=matrix)
    assert isclose(ret.vectors.std(), 0.8668510720176622)
    first = ret.vectors[0, 0]
    np.testing.assert_allclose(first, [-0.712261, 1.378551, -0.213198],
                               atol=1e-06)


@pytest.vasporicache_available  # type: ignore
def test_symmetrize(vasp_orientations):
    ret = vasp_orientations.symmetrize(sym_group='m-3m')
    assert isclose(ret.vectors.std(), 0.8668511628167984)
    first = ret.vectors[0, 0]
    np.testing.assert_allclose(first, [1.330221, -0.679495, -0.471138],
                               atol=1e-06)


@pytest.vasporicache_available  # type: ignore
def test_normalize_symmetrize(vasp_orientations):
    ret = vasp_orientations.normalize().symmetrize(sym_group='m-3m')
    assert isclose(ret.vectors.std(), 0.577350269189626)
    first = ret.vectors[0, 0]
    np.testing.assert_allclose(first, [0.849297, -0.433833, -0.300804],
                               atol=1e-06)


@pytest.vasporicache_available  # type: ignore
def test_normalize_conventional_symmetrize(vasp_orientations):
    ret = vasp_orientations.normalize().transform(matrix=matrix).symmetrize(
        sym_group='m-3m')
    assert isclose(ret.vectors.std(), 0.577350269189626)
    first = ret.vectors[0, 0]
    np.testing.assert_allclose(first, [0.136119, -0.880154, 0.454753],
                               atol=1e-06)


@pytest.vasporicache_available  # type: ignore
def test_conventional_normalize_symmetrize(vasp_orientations):
    ret = vasp_orientations.transform(matrix=matrix).normalize().symmetrize(
        sym_group='m-3m')
    assert isclose(ret.vectors.std(), 0.577350269189626)
    first = ret.vectors[0, 0]
    np.testing.assert_allclose(first, [0.136119, -0.880154, 0.454753],
                               atol=1e-06)


@pytest.vasporicache_available  # type: ignore
def test_fractional_coordinates(vasp_orientations):
    direction = vasp_orientations._fractional_directions(
        vasp_orientations._distances)

    assert isclose(direction.mean(), -2.8363494021277384e-05)
    assert isclose(direction[3, 12, 2], -0.10902666)


@pytest.vasporicache_available  # type: ignore
def test_matching_matrix(vasp_orientations):
    distance = np.array([[1.0, 2.0, 1.5, 0.5, 3.0], [2.0, 1.0, 0.5, 1.5, 3.0]])
    frac_coord_cent = np.array([[[0.3, 0.11, 0.78], [1.6, 1.0, 2.3]]])
    matching_matrix = vasp_orientations._matching_matrix(
        distance, frac_coord_cent)

    assert isclose(matching_matrix[-1, -1], 2)
    assert isclose(matching_matrix.sum(), 20)


def test_calculate_spherical_areas():
    area_grid = calculate_spherical_areas((30, 75), 1)

    assert isclose(area_grid[0, 1], 3.7304874810631827e-20)
    assert isclose(area_grid.mean(), 0.00018727792392184294)
