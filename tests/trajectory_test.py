import numpy as np
from gemdat.trajectory import Trajectory
from pymatgen.core import Lattice, Species


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
    assert trajectory.time_step == t2.time_step

    np.testing.assert_allclose(trajectory.lattice, t2.lattice)
    np.testing.assert_allclose(trajectory.base_positions, t2.base_positions)
    np.testing.assert_allclose(trajectory.coords, t2.coords)


def test_displacements_property(trajectory):
    trajectory.to_positions()

    np.testing.assert_allclose(trajectory.displacements, [
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.3, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ])

    assert trajectory.coords_are_displacement


def test_positions_property(trajectory):
    trajectory.to_displacements()

    np.testing.assert_allclose(trajectory.positions, [
        [[0.2, 0.0, 0.0], [0.0, 0.0, 0.5]],
        [[0.4, 0.0, 0.0], [0.0, 0.0, 0.5]],
        [[0.6, 0.0, 0.0], [0.0, 0.0, 0.5]],
        [[0.8, 0.0, 0.0], [0.0, 0.0, 0.5]],
        [[1.1, 0.0, 0.0], [0.0, 0.0, 0.5]],
    ])

    assert not trajectory.coords_are_displacement
