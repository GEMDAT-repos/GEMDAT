from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pymatgen.core import Lattice, PeriodicSite, Structure
from pymatgen.io.vasp import VolumetricData

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
    ((0.0, 0.0, 0.0), (0, 0, 0)),
    ((0.5, 0.5, 0.5), (1, 2, 2)),
    ((1.0, 1.0, 1.0), (3, 4, 5)),
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
    ((0, 0, 0), (1.0, 1.0, 1.0)),
    ((3, 4, 5), (7.0, 9.0, 11.0)),
)


@pytest.mark.parametrize('voxel,expected', TEST_DATA_VOX_TO_CART)
def test_volume_vox_to_cart(volume, voxel, expected):
    ret = volume.voxel_to_cart_coords(voxel)
    assert_allclose(ret, expected)


def test_voxel_size(volume):
    assert_allclose(volume.voxel_size, (2.0, 2.0, 2.0))


def test_dedup_pbc_peaks_merges_boundary_split():
    """Two peaks that only coincide across a periodic boundary collapse to
    one."""
    latt = Lattice.from_parameters(10, 10, 10, 90, 90, 90)
    vol = Volume(data=np.ones((20, 20, 20)), lattice=latt)

    # Same site split across the x=0 face (voxel 1 and voxel 19 of 20), plus an
    # unrelated interior peak that must be kept.
    peaks = np.array([[1, 10, 10], [19, 10, 10], [10, 5, 5]])
    out = vol._dedup_pbc_peaks(peaks, tol=1.0)

    assert len(out) == 2
    # the interior peak survives untouched
    assert any(np.array_equal(p, [10, 5, 5]) for p in out)


def test_dedup_pbc_peaks_keeps_genuinely_close_interior_peaks():
    """Interior peaks that are close in direct space are not merged."""
    latt = Lattice.from_parameters(10, 10, 10, 90, 90, 90)
    vol = Volume(data=np.ones((20, 20, 20)), lattice=latt)

    # Two adjacent interior peaks ~0.5 Å apart (direct distance < tol): a real,
    # if dense, pair of sites — not a wrap artefact, so both are kept.
    peaks = np.array([[10, 10, 10], [11, 10, 10]])
    out = vol._dedup_pbc_peaks(peaks, tol=1.0)

    assert len(out) == 2


def test_to_structure_snap_to_lower():
    """`snap_to_lower` moves an on-face site to the lower end; default keeps
    it."""
    latt = Lattice.from_parameters(10, 10, 10, 90, 90, 90)
    # A single blob straddling the x=0 face, with most of its mass on the high
    # side (voxels 8, 9, 0), so its centroid lands just below the upper face.
    data = np.zeros((10, 10, 10))
    for x in (8, 9, 0):
        data[x, 5, 5] = 1.0
    vol = Volume(data=data, lattice=latt)
    peaks = np.array([[9, 5, 5]])

    default = vol.to_structure(peaks=peaks, background_level=0.1)
    snapped = vol.to_structure(peaks=peaks, background_level=0.1, snap_to_lower=True)

    assert len(default) == len(snapped) == 1
    # default: reported near the upper boundary; snapped: pulled to ~0
    assert default.frac_coords[0, 0] > 0.9
    assert snapped.frac_coords[0, 0] == 0.0


def test_to_vasp_volume(volume):
    structure = Structure(
        lattice=volume.lattice,
        coords=[(0, 0, 0), (0.5, 0.5, 0.5)],
        species=['Si', 'Si'],
    )
    volume2 = Volume(data=volume.data, lattice=volume.lattice, label='free_energy')
    vol_data = volume.to_vasp_volume(structure=structure, other=[volume2])
    assert isinstance(vol_data, VolumetricData)
    assert set(vol_data.data.keys()) == {'total', 'free_energy'}
