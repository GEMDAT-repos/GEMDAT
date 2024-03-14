from __future__ import annotations

from math import isclose

import pytest


@pytest.vasprotocache_available  # type: ignore
def test_Oh_point_group(Oh_sym_matrices):
    assert Oh_sym_matrices.shape == (3, 3, 48)
    assert (Oh_sym_matrices * Oh_sym_matrices).sum() == 144


@pytest.vasprotocache_available  # type: ignore
def test_direct_coordinates(vasp_orientations):
    dc = vasp_orientations.get_unit_vectors_traj(normalize=False)
    assert isclose(dc.mean(), -0.0005719846079221715)


@pytest.vasprotocache_available  # type: ignore
def test_conventional_coordinates(vasp_orientations):
    cf = vasp_orientations.get_conventional_coordinates(normalize=True)
    cf_spheric = vasp_orientations.cartesian_to_spherical(direct_cart=cf,
                                                          degrees=True)

    assert isclose(cf.mean(), -0.00020323016902595493)
    assert isclose(cf_spheric.mean(), 0.07124811161160129)


@pytest.vasprotocache_available  # type: ignore
def test_symmetrize_traj(vasp_orientations, Oh_sym_matrices):
    sym_t = vasp_orientations.get_symmetric_traj(Oh_sym_matrices[:, :, :6])

    assert isclose(sym_t.mean(), -0.00020323016902595498)
