from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pymatgen.core import Lattice, PeriodicSite

from gemdat import Volume


@pytest.fixture(scope='module')
def volume():
    data = np.arange(60).reshape(3, 4, 5)
    latt = Lattice.from_parameters(6, 8, 10, 90, 90, 90)

    return Volume(data=data, lattice=latt)


TEST_DATA_VOX_TO_FRAC = (
    # voxel, expected
    ((-0.5, -0.5, -0.5), (0, 0, 0)),
    ((0, 0, 0), (1 / 6, 1 / 8, 1 / 10)),
    (np.array((3, 4, 5)) / 2, (4 / 6, 5 / 8, 6 / 10)),
    (np.array((3, 4, 5)) - 0.5, (1, 1, 1)),
    ((3, 4, 5), (7 / 6, 9 / 8, 11 / 10)),
)


@pytest.mark.parametrize('voxel,expected', TEST_DATA_VOX_TO_FRAC)
def test_volume_voxel_to_frac(volume, voxel, expected):
    ret = volume.voxel_to_frac_coords(voxel)
    assert_allclose(ret, expected)


TEST_DATA_FRAC_TO_VOX = (
    # coord, expected
    ((0., 0., 0.), (0, 0, 0)),
    ((0.5, 0.5, 0.5), (1, 2, 2)),
    ((1., 1., 1.), (3, 4, 5)),
)


@pytest.mark.parametrize('coord,expected', TEST_DATA_FRAC_TO_VOX)
def test_volume_frac_to_vox(volume, coord, expected):
    ret = volume.frac_coords_to_voxel(coord)
    assert_allclose(ret, expected)


@pytest.mark.parametrize('coord,expected', TEST_DATA_FRAC_TO_VOX)
def test_volume_site_to_vox(volume, coord, expected):
    site = PeriodicSite('X', coord, volume.lattice)

    ret = volume.site_to_voxel(site)
    assert_allclose(ret, expected)


TEST_DATA_VOX_TO_CART = (
    # voxel, expected
    ((-0.5, -0.5, -0.5), (0, 0, 0)),
    ((0, 0, 0), (1., 1., 1.)),
    ((3, 4, 5), (7., 9., 11.)),
)


@pytest.mark.parametrize('voxel,expected', TEST_DATA_VOX_TO_CART)
def test_volume_vox_to_cart(volume, voxel, expected):
    ret = volume.voxel_to_cart_coords(voxel)
    assert_allclose(ret, expected)


def test_voxel_size(volume):
    assert_allclose(volume.voxel_size, (2., 2., 2.))
