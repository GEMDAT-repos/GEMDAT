from __future__ import annotations

from math import isclose

import numpy as np
import pytest

from gemdat.rotations import calculate_spherical_areas
from gemdat.utils import cartesian_to_spherical


@pytest.vasprotocache_available  # type: ignore
def test_direct_coordinates(vasp_orientations):
    dc = vasp_orientations.get_unit_vectors_traj()
    assert isclose(dc.mean(), -0.0005719846079221715)


@pytest.vasprotocache_available  # type: ignore
def test_conventional_coordinates(vasp_orientations):
    cf = vasp_orientations.get_conventional_coordinates()
    cf_spheric = cartesian_to_spherical(cf, degrees=True)

    assert isclose(cf.mean(), -0.00039676020882101193)
    assert isclose(cf_spheric.mean(), 0.23810303372936106)


@pytest.vasprotocache_available  # type: ignore
def test_symmetrize_traj(vasp_orientations):
    vasp_orientations.set_symmetry_operations(sym_group='m-3m')
    sym_t = vasp_orientations.get_symmetric_traj()

    assert isclose((sym_t * sym_t).mean(), 0.7514309384768313)


@pytest.vasprotocache_available  # type: ignore
def test_fractional_coordinates(vasp_orientations):
    direction = vasp_orientations.fractional_directions(
        vasp_orientations._distances)

    assert isclose(direction.mean(), -2.8363494021277384e-05)
    assert isclose(direction[3, 12, 2], -0.10902666)


@pytest.vasprotocache_available  # type: ignore
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
