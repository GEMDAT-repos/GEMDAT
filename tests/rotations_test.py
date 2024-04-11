from __future__ import annotations

from math import isclose

import numpy as np
from pymatgen.core import Species

from gemdat.orientations import (
    Orientations,
    calculate_spherical_areas,
)
from gemdat.utils import autocorrelation


def test_orientations_init(trajectory):
    orientations = Orientations(trajectory=trajectory,
                                center_type='B',
                                satellite_type='Si',
                                nr_central_atoms=1)

    assert isinstance(orientations, Orientations)
    assert orientations.center_type == 'B'
    assert orientations.satellite_type == 'Si'
    assert orientations.nr_central_atoms == 1


def test_normalize(trajectory):
    orientations = Orientations(trajectory=trajectory,
                                center_type='B',
                                satellite_type='Si',
                                nr_central_atoms=1,
                                in_vectors=np.array([[1, 2, 2], [2, 2, 1]],
                                                    dtype=float))
    ret = orientations.normalize()
    assert np.allclose(
        ret.vectors, np.array([[1 / 3, 2 / 3, 2 / 3], [2 / 3, 2 / 3, 1 / 3]]))


def test_conventional(trajectory):
    orientations = Orientations(trajectory=trajectory,
                                center_type='B',
                                satellite_type='Si',
                                nr_central_atoms=1,
                                in_vectors=np.array(
                                    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                    dtype=float))
    matrix = np.eye(3) * [1, 2, 3]
    ret = orientations.transform(matrix=matrix)
    assert np.allclose(ret.vectors, np.array([[1, 0, 0], [0, 2, 0], [0, 0,
                                                                     3]]))


def test_symmetrize(trajectory):
    orientations = Orientations(trajectory=trajectory,
                                center_type='B',
                                satellite_type='Si',
                                nr_central_atoms=1,
                                in_vectors=np.array([[[1, 0, 0]], [[0, 1, 0]]],
                                                    dtype=float))
    sym_ops = np.array([[0, -1, 0], [1, 0, 0], [0, 0, -1]])
    ret = orientations.symmetrize(sym_ops=sym_ops)
    assert np.allclose(
        ret.vectors,
        np.array([[[0., 1., 0.], [-1., 0., 0.], [0., 0., -1.]],
                  [[0., 1., 0.], [-1., 0., 0.], [0., 0., -1.]]]))


def test_orientations(orientations):
    assert orientations._time_step == 1
    assert orientations._trajectory_cent.species == [Species('B')]
    assert orientations._trajectory_sat.species == [Species('Si')]


def test_fractional_coordinates(orientations):
    frac_coord_cent, frac_coord_sat = orientations._fractional_coordinates()
    assert isinstance(frac_coord_cent, np.ndarray)
    assert isinstance(frac_coord_sat, np.ndarray)


def test_distances(orientations):
    distances = orientations._distances
    assert isinstance(distances, np.ndarray)


def test_calculate_spherical_areas():
    shape = (10, 10)
    areas = calculate_spherical_areas(shape)
    assert isclose(areas.mean(), 0.00017275712347752164)
    assert isinstance(areas, np.ndarray)
    assert areas.shape == shape


def test_autocorrelation(trajectory):
    autocorr = autocorrelation(trajectory.positions)
    assert isinstance(autocorr, np.ndarray)
    assert isclose(autocorr.mean(), 0.8142314269325723)
    assert autocorr.shape == (trajectory.positions.shape[1],
                              trajectory.positions.shape[0])
