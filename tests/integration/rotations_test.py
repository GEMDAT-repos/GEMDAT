from __future__ import annotations

from math import isclose

import numpy as np
import pytest

from gemdat.rotations import calculate_spherical_areas

TEST_TRANSFORM = (
    # transforms, expected
    (None, (0.8668509741071079, [0.47113781, 0.67949494, -1.33022098])),
    (['normalize'], (0.5773501929401034, [0.300804, 0.433833, -0.849297])),
    (['conventional'], (0.8668510720176622, [-0.712261, 1.378551, -0.213198])),
    (['symmetrize'], (0.8668511628167984, [1.330221, -0.679495, -0.471138])),
    (['normalize',
      'symmetrize'], (0.577350269189626, [0.849297, -0.433833, -0.300804])),
    (['normalize', 'conventional',
      'symmetrize'], (0.577350269189626, [0.136119, -0.880154, 0.454753])),
    (['conventional', 'normalize',
      'symmetrize'], (0.577350269189626, [0.136119, -0.880154, 0.454753])),
)


@pytest.vasprotocache_available  # type: ignore
@pytest.mark.parametrize('transforms,expected', TEST_TRANSFORM)
def test_transforms(vasp_orientations, transforms, expected):
    expected_std, expected_first = expected
    transformed_traj = vasp_orientations.transform(transforms)

    assert isclose(transformed_traj.std(), expected_std)

    first = transformed_traj[0, 0]
    np.testing.assert_allclose(first, expected_first, atol=1e-06)


@pytest.vasprotocache_available  # type: ignore
def test_fractional_coordinates(vasp_orientations):
    direction = vasp_orientations._fractional_directions(
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
