from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from pymatgen.core import Structure

from gemdat.utils import (
    bfill,
    cartesian_to_spherical,
    ffill,
    integer_remap,
    meanfreq,
    remove_partial_occupancies_from_structure,
    require_constant_lattice,
)


@pytest.fixture
def arr():
    return np.array(
        [
            [5, -1, -1, 7, 2],
            [3, -1, 1, 8, -1],
            [4, 9, 6, -1, -1],
        ]
    )


def test_ffill(arr):
    ret = ffill(arr)
    expected = np.array(
        [
            [5, 5, 5, 7, 2],
            [3, 3, 1, 8, 8],
            [4, 9, 6, 6, 6],
        ]
    )

    assert_equal(ret, expected)


def test_bfill(arr):
    ret = bfill(arr)
    expected = np.array(
        [
            [5, 7, 7, 7, 2],
            [3, 1, 1, 8, -1],
            [4, 9, 6, -1, -1],
        ]
    )

    assert_equal(ret, expected)


def test_ffill_axis0(arr):
    ret = ffill(arr, axis=0)
    expected = np.array(
        [
            [5, -1, -1, 7, 2],
            [3, -1, 1, 8, 2],
            [4, 9, 6, 8, 2],
        ]
    )

    assert_equal(ret, expected)


def test_bfill_axis0(arr):
    ret = bfill(arr, axis=0)
    expected = np.array(
        [
            [5, 9, 1, 7, 2],
            [3, 9, 1, 8, -1],
            [4, 9, 6, -1, -1],
        ]
    )

    assert_equal(ret, expected)


def test_integer_remap():
    a = np.array([4, 2, 1, 3])
    key = np.array([10, 20, 30, 40])
    ret = integer_remap(a, key=key)
    assert_equal(ret, a * 10)


def test_meanfreq_single_timestep():
    x = np.sin(np.linspace(0, 1, 6))
    ret = meanfreq(x)

    expected = np.array([[0.2303359]])

    assert_allclose(ret, expected)


def test_meanfreq():
    x = np.array(
        [
            np.sin(np.linspace(0, 1, 6)),
            np.sin(np.linspace(0, 2, 6)),
            np.sin(np.linspace(0, 3, 6)),
        ]
    )
    ret = meanfreq(x)

    expected = np.array([[0.2303359], [0.21308077], [0.17074241]])

    assert_allclose(ret, expected)


def test_cartesian_to_spherical():
    x = np.array([1, 0])
    y = np.array([0, 1])
    z = np.array([1, 1])
    xyz = np.expand_dims(np.stack((x, y, z), axis=-1), axis=0)
    # test in radians
    ret = cartesian_to_spherical(xyz, degrees=False)
    expected = np.array(
        [
            [
                [0, np.arccos(1 / np.sqrt(2)), np.sqrt(2)],
                [np.pi / 2, np.arccos(1 / np.sqrt(2)), np.sqrt(2)],
            ]
        ]
    )
    assert_allclose(ret, expected, rtol=1e-5)

    ret = cartesian_to_spherical(xyz, degrees=True)
    expected = np.array(
        [
            [
                [0, np.degrees(np.arccos(1 / np.sqrt(2))), np.sqrt(2)],
                [
                    np.degrees(np.pi / 2),
                    np.degrees(np.arccos(1 / np.sqrt(2))),
                    np.sqrt(2),
                ],
            ]
        ]
    )
    assert_allclose(ret, expected, rtol=1e-5)


def test_remove_partial_occupancies_from_structure():
    structure = Structure(
        lattice=np.eye(3) * 10,
        coords=[(0, 0, 0), (0.5, 0.5, 0.5)],
        species=[{'Si': 0.5, 'Ge': 0.5}, {'Ge': 0.5}],
        labels=['A', 'B'],
    )
    assert not structure.is_ordered

    new_structure = remove_partial_occupancies_from_structure(structure)
    assert new_structure.is_ordered
    assert len(new_structure) == 2
    assert new_structure.labels == structure.labels


class _FakeTrajectory:
    def __init__(self, constant_lattice: bool):
        self.constant_lattice = constant_lattice


def test_require_constant_lattice_from_trajectory_argument():
    @require_constant_lattice
    def analyse(trajectory, factor=2):
        return factor

    assert analyse(_FakeTrajectory(True)) == 2

    with pytest.raises(NotImplementedError, match='variable lattice'):
        analyse(_FakeTrajectory(False))


def test_require_constant_lattice_from_self():
    class Analysis:
        def __init__(self, trajectory):
            self.trajectory = trajectory

        @require_constant_lattice
        def run(self):
            return 'ok'

    assert Analysis(_FakeTrajectory(True)).run() == 'ok'

    with pytest.raises(NotImplementedError, match='variable lattice'):
        Analysis(_FakeTrajectory(False)).run()


def test_require_constant_lattice_from_flag():
    # Readers take the flag directly, before a trajectory exists. The default
    # must be honoured when the argument is not passed.
    @require_constant_lattice
    def read(path, constant_lattice=True):
        return path

    assert read('a.xyz') == 'a.xyz'
    assert read('a.xyz', constant_lattice=True) == 'a.xyz'

    with pytest.raises(NotImplementedError, match='variable lattice'):
        read('a.xyz', constant_lattice=False)


def test_require_constant_lattice_names_the_function():
    @require_constant_lattice
    def some_analysis(trajectory):
        return None

    with pytest.raises(NotImplementedError, match='some_analysis'):
        some_analysis(_FakeTrajectory(False))


def test_require_constant_lattice_without_a_source():
    @require_constant_lattice
    def nothing_to_check(value):
        return value

    with pytest.raises(TypeError, match='constant_lattice'):
        nothing_to_check(1)


def test_require_constant_lattice_from_a_wrapper_object():
    # rdf.radial_distribution takes `transitions`, not a trajectory.
    class Transitions:
        def __init__(self, trajectory):
            self.trajectory = trajectory

    @require_constant_lattice
    def analyse(*, transitions):
        return 'ok'

    assert analyse(transitions=Transitions(_FakeTrajectory(True))) == 'ok'

    with pytest.raises(NotImplementedError, match='variable lattice'):
        analyse(transitions=Transitions(_FakeTrajectory(False)))


def test_require_constant_lattice_skips_unrelated_arguments():
    # `sites` comes first in _calculate_atom_states and carries no lattice flag.
    @require_constant_lattice
    def analyse(sites, trajectory):
        return 'ok'

    assert analyse(object(), _FakeTrajectory(True)) == 'ok'

    with pytest.raises(NotImplementedError, match='variable lattice'):
        analyse(object(), _FakeTrajectory(False))


def test_require_constant_lattice_runs_on_cache_hit():
    # The guard must sit above weak_lru_cache, or a cached call skips it.
    from gemdat.caching import weak_lru_cache

    class Analysis:
        def __init__(self, trajectory):
            self.trajectory = trajectory

        @require_constant_lattice
        @weak_lru_cache()
        def run(self, dimensions):
            return dimensions

    analysis = Analysis(_FakeTrajectory(True))
    assert analysis.run(3) == 3

    analysis.trajectory.constant_lattice = False
    with pytest.raises(NotImplementedError, match='variable lattice'):
        analysis.run(3)
