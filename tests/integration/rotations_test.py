from __future__ import annotations

from math import isclose

import pytest

from gemdat.rotations import cartesian_to_spherical


@pytest.vasprotocache_available  # type: ignore
def test_Oh_point_group(Oh_sym_matrices):
    assert Oh_sym_matrices.shape == (3, 3, 48)
    assert (Oh_sym_matrices * Oh_sym_matrices).sum() == 144


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
def test_symmetrize_traj(vasp_orientations, Oh_sym_matrices):
    vasp_orientations.set_symmetry_operations(Oh_sym_matrices[:, :, :6])
    sym_t = vasp_orientations.get_symmetric_traj()

    assert isclose(sym_t.mean(), -0.000396760208821012)
