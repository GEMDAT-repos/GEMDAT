from __future__ import annotations

from math import isclose

import numpy as np
from numpy.testing import assert_allclose
from pymatgen.core import Species

from gemdat.orientations import (
    Orientations,
    calculate_spherical_areas,
)
from gemdat.trajectory import Trajectory
from gemdat.utils import fft_autocorrelation


def test_orientations_init(trajectory):
    orientations = Orientations(trajectory=trajectory, center_type='B', satellite_type='Si')

    assert isinstance(orientations, Orientations)
    assert orientations.center_type == 'B'
    assert orientations.satellite_type == 'Si'
    assert orientations.trajectory == trajectory


def test_normalize(trajectory):
    orientations = Orientations(
        trajectory=trajectory,
        center_type='B',
        satellite_type='Si',
        in_vectors=np.array([[1, 2, 2], [2, 2, 1]], dtype=float),
    )
    ret = orientations.normalize()
    assert_allclose(ret.vectors, np.array([[1 / 3, 2 / 3, 2 / 3], [2 / 3, 2 / 3, 1 / 3]]))


def test_conventional(trajectory):
    orientations = Orientations(
        trajectory=trajectory,
        center_type='B',
        satellite_type='Si',
        in_vectors=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
    )
    matrix = np.eye(3) * [1, 2, 3]
    ret = orientations.transform(matrix=matrix)
    assert_allclose(ret.vectors, np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]))


def test_symmetrize(trajectory):
    orientations = Orientations(
        trajectory=trajectory,
        center_type='B',
        satellite_type='Si',
        in_vectors=np.array([[[1, 0, 0]], [[0, 1, 0]]], dtype=float),
    )
    sym_ops = np.array([[0, -1, 0], [1, 0, 0], [0, 0, -1]])
    ret = orientations.symmetrize(sym_ops=sym_ops)
    assert_allclose(
        ret.vectors,
        np.array(
            [
                [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
                [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
            ]
        ),
    )


def test_orientations(orientations):
    assert orientations._time_step == 1
    assert orientations._trajectory_cent.species == [Species('B')]
    assert orientations._trajectory_sat.species == [Species('Si')]


def test_distances(orientations):
    distances = orientations._distances
    assert isinstance(distances, np.ndarray)


def test_calculate_spherical_areas():
    shape = (10, 10)
    areas = calculate_spherical_areas(shape)
    assert isclose(areas.mean(), 0.00017275712347752164)
    assert isinstance(areas, np.ndarray)
    assert areas.shape == shape


def test_fft_autocorrelation(trajectory):
    autocorr = fft_autocorrelation(trajectory.positions)
    assert isinstance(autocorr, np.ndarray)
    assert isclose(autocorr.mean(), 0.8142314269325723)
    assert autocorr.shape == (
        trajectory.positions.shape[1],
        trajectory.positions.shape[0],
    )


def _molecule_trajectory(lattice, *, constant_lattice):
    """Single central S with four O ligands, fractional coords frozen in
    time."""
    offsets = np.array(
        [
            [0.08, 0.08, 0.08],
            [-0.08, -0.08, 0.08],
            [0.08, -0.08, -0.08],
            [-0.08, 0.08, -0.08],
        ]
    )
    frame = np.vstack([[0.5, 0.5, 0.5], 0.5 + offsets])

    return Trajectory(
        species=[Species('S')] + [Species('O')] * 4,
        coords=np.tile(frame, (4, 1, 1)),
        lattice=lattice,
        constant_lattice=constant_lattice,
        metadata={'temperature': 300},
        time_step=1,
    )


def test_orientations_variable_lattice_uses_per_frame_cell():
    # issue #394: vectors were built with a single lattice, which is wrong (and
    # since get_lattice() now raises, impossible) for an NPT cell.
    scales = np.array([1.0, 1.02, 1.04, 1.06])
    lattices = np.array([np.eye(3) * 6.0 * scale for scale in scales])

    trajectory = _molecule_trajectory(lattices, constant_lattice=False)
    orientations = Orientations(trajectory=trajectory, center_type='S', satellite_type='O')

    assert orientations.vectors.shape == (4, 4, 3)

    # Fractional coords are frozen, so bond lengths must scale with the cell.
    norms = np.linalg.norm(orientations.vectors, axis=-1)
    assert_allclose(norms / norms[0], np.broadcast_to(scales[:, None], norms.shape))


def test_orientations_variable_lattice_matches_constant():
    # With every frame's lattice identical, both paths must agree exactly.
    lattice = np.diag([6.0, 7.0, 8.0])

    const = Orientations(
        trajectory=_molecule_trajectory(lattice, constant_lattice=True),
        center_type='S',
        satellite_type='O',
    )
    var = Orientations(
        trajectory=_molecule_trajectory(np.array([lattice] * 4), constant_lattice=False),
        center_type='S',
        satellite_type='O',
    )

    assert_allclose(var.vectors, const.vectors)
