from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pymatgen.core import Lattice, PeriodicSite
from pymatgen.symmetry.groups import SpaceGroup

from gemdat.shape import ShapeAnalyzer, ShapeData


@pytest.fixture
def site():
    latt = Lattice.from_parameters(10, 10, 10, 90, 90, 90)
    site = PeriodicSite('Si', (0.1, 0.2, 0.3), latt, label='A')
    return site


@pytest.fixture
def shape(site):
    coords = np.array([
        [0, 3, 4],
        [5, 0, 12],
        [8, 15, 0],
    ])
    return ShapeData(site=site, coords=coords, radius=20)


def test_distances(shape):
    dists = shape.distances()
    assert_allclose(dists, [5, 13, 17])


def test_xyz(shape):
    assert_allclose(shape.x, [0, 5, 8])
    assert_allclose(shape.y, [3, 0, 15])
    assert_allclose(shape.z, [4, 12, 0])


def test_centroid(shape):
    assert_allclose(shape.centroid(), 5.742857, 7.714286, 5.028571)


@pytest.fixture
def shape_analyzer(site):
    spgr = SpaceGroup('P-1')
    return ShapeAnalyzer(lattice=site.lattice, sites=[site], spacegroup=spgr)


def test_shape_analyzer(shape_analyzer):
    assert shape_analyzer.lattice
    assert shape_analyzer.spacegroup
    assert len(shape_analyzer.sites) == 1


def test_shape_analyzer_analyze_positions(shape_analyzer):
    shapes = shape_analyzer.analyze_positions(
        np.array([
            (0.2, 0.2, 0.2),
            (0.8, 0.8, 0.8),
        ]),
        radius=2,
    )

    assert len(shapes) == 1

    shape = shapes[0]

    assert isinstance(shape, ShapeData)

    assert shape.name == 'A'
    assert shape.radius == 2

    assert_allclose(shape.x, 1, atol=1e-07)
    assert_allclose(shape.y, 0, atol=1e-07)
    assert_allclose(shape.z, -1, atol=1e-07)

    assert_allclose(shape.distances(), 2**0.5)
    assert_allclose(shape.centroid(), (1, 0, -1), atol=1e-07)


def test_shape_analyzer_to_structure(shape_analyzer):
    structure = shape_analyzer.to_structure()

    assert structure.lattice.parameters == (10, 10, 10, 90, 90, 90)
    assert len(structure) == 2
    assert [sp.name for sp in structure.species] == ['Si', 'Si']

    assert structure.labels == ['A', 'A']


def test_shape_analyzer_shift_sites(shape_analyzer):
    shifted1 = shape_analyzer.shift_sites(
        vectors=[(0.2, 0.2, 0.2)],
        coords_are_cartesian=False,
    )
    site = shifted1.sites[0]
    assert_allclose(site.frac_coords, (0.3, 0.4, 0.5))

    shifted2 = shifted1.shift_sites(
        vectors=[(-2, -2, -2)],
        coords_are_cartesian=True,
    )
    site = shifted2.sites[0]
    assert_allclose(site.frac_coords, (0.1, 0.2, 0.3))


def test_shape_analyzer_optimize_sites(shape_analyzer):
    site = shape_analyzer.sites[0]
    assert_allclose(site.frac_coords, (0.1, 0.2, 0.3))

    shape = ShapeData(site=site, coords=np.array([[2, 2, 2]]), radius=1)
    shifted = shape_analyzer.optimize_sites((shape, ))

    site = shifted.sites[0]
    assert_allclose(site.frac_coords, (0.3, 0.4, 0.5))
