import numpy as np
import pytest
from gemdat.trajectory import Trajectory
from pymatgen.core import Lattice, Species


@pytest.fixture
def trajectory():
    coords = np.array([
        [[.0, .0, .0], [.0, .0, .5]],
        [[.2, .0, .0], [.0, .0, .5]],
        [[.4, .0, .0], [.0, .0, .5]],
        [[.6, .0, .0], [.0, .0, .5]],
        [[.8, .0, .0], [.0, .0, .5]],
    ])

    return Trajectory(species=[Species('B'), Species('C')],
                      coords=coords,
                      lattice=np.eye(3),
                      metadata={'temperature': 123})


def test_trajectory(trajectory):
    assert isinstance(trajectory, Trajectory)
    assert trajectory.species == [Species('B'), Species('C')]
    assert trajectory.coords.shape == (5, 2, 3)
    assert trajectory.metadata == {'temperature': 123}


def test_slice(trajectory):
    sliced = trajectory[2:]

    assert isinstance(sliced, Trajectory)
    assert sliced.species == trajectory.species
    assert sliced.coords.shape == (3, 2, 3)
    assert sliced.metadata == trajectory.metadata


def test_filter(trajectory):
    t = trajectory.filter('C')
    assert t.species == [Species('C')]
    assert np.all(t.coords == [.0, .0, .5])


def test_get_lattice(trajectory):
    lattice = trajectory.get_lattice()
    expected_lattice = Lattice(np.eye(3))

    assert isinstance(lattice, Lattice)
    assert lattice == expected_lattice


def test_caching(trajectory, tmpdir):
    cachefile = tmpdir / 'trajectory.cache'
    trajectory.to_cache(cachefile)

    assert cachefile.exists()

    t2 = Trajectory.from_cache(cachefile)

    assert trajectory.species == t2.species
    assert trajectory.metadata == t2.metadata

    np.testing.assert_array_equal(trajectory.lattice, t2.lattice)
    np.testing.assert_array_equal(trajectory.base_positions, t2.base_positions)
    np.testing.assert_array_equal(trajectory.coords, t2.coords)
