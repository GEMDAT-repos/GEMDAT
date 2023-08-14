import numpy as np
from gemdat.trajectory import Trajectory
from numpy.testing import assert_allclose
from pymatgen.core import Lattice, Species


def test_trajectory(trajectory):
    assert isinstance(trajectory, Trajectory)
    assert trajectory.species == [Species('B'), Species('C')]
    assert trajectory.positions.shape == (5, 2, 3)
    assert trajectory.metadata == {'temperature': 123}


def test_slice(trajectory):
    sliced = trajectory[2:]

    assert isinstance(sliced, Trajectory)
    assert sliced.species == trajectory.species
    assert sliced.positions.shape == (3, 2, 3)
    assert sliced.metadata == trajectory.metadata


def test_filter(trajectory):
    t = trajectory.filter('C')
    assert t.species == [Species('C')]
    assert np.all(t.positions == [.0, .0, .5])


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

    assert_allclose(trajectory.lattice, t2.lattice)
    assert_allclose(trajectory.base_positions, t2.base_positions)
    assert_allclose(trajectory.positions, t2.positions)


def test_displacements_property(trajectory):
    trajectory.to_positions()

    assert_allclose(trajectory.displacements, [
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.3, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ])

    assert trajectory.coords_are_displacement


def test_positions_property(trajectory):
    trajectory.to_displacements()

    assert_allclose(trajectory.positions, [
        [[0.2, 0.0, 0.0], [0.0, 0.0, 0.5]],
        [[0.4, 0.0, 0.0], [0.0, 0.0, 0.5]],
        [[0.6, 0.0, 0.0], [0.0, 0.0, 0.5]],
        [[0.8, 0.0, 0.0], [0.0, 0.0, 0.5]],
        [[1.1, 0.0, 0.0], [0.0, 0.0, 0.5]],
    ])

    assert not trajectory.coords_are_displacement


def test_drift_correction(trajectory):
    drift = trajectory.drift(fixed_species='B')
    assert drift.shape == (5, 1, 3)
    global_drift = np.mean(drift, axis=0)
    assert_allclose(global_drift, [[0.18, 0.0, 0.0]])

    t2 = trajectory.apply_drift_correction(fixed_species='B')
    global_drift2 = np.mean(t2.drift(fixed_species='B'), axis=0)

    # drift must now be effectively removed
    assert_allclose(global_drift2, [[0.0, 0.0, 0.0]])


def test_total_distances(trajectory):
    distances = trajectory.total_distances()
    assert_allclose(distances, [
        [0.0, 0.2, 0.4, 0.6, 0.9],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ])


def test_total_displacements(trajectory):
    displacements = trajectory.total_displacements
    assert_allclose(displacements, [
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.4, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.6, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.9, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ])
