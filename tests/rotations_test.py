from __future__ import annotations

from math import isclose

import numpy as np
from pymatgen.core import Species

from gemdat.rotations import (
    Conventional,
    Normalize,
    Orientations,
    Symmetrize,
    calculate_spherical_areas,
    mean_squared_angular_displacement,
)


def test_orientations_init(trajectory):
    orientations = Orientations(trajectory=trajectory,
                                center_type='B',
                                satellite_type='Si',
                                nr_central_atoms=1)

    assert isinstance(orientations, Orientations)
    assert orientations.center_type == 'B'
    assert orientations.satellite_type == 'Si'
    assert orientations.nr_central_atoms == 1


def test_set_symmetry_operations(trajectory):
    orientations = Orientations(trajectory=trajectory,
                                center_type='B',
                                satellite_type='Si',
                                nr_central_atoms=1)

    sym_matrix = np.array([[[0, -1, 0], [1, 0, 0], [0, 0, -1]]])
    orientations.set_symmetry_operations(explicit_sym=sym_matrix)

    assert np.array_equal(orientations.sym_matrix, sym_matrix)


def test_normalize(trajectory):
    orientation = Orientations(trajectory=trajectory,
                               center_type='B',
                               satellite_type='Si',
                               nr_central_atoms=1)
    orientation.transformed_trajectory = np.array([[1, 2, 2], [2, 2, 1]],
                                                  dtype=float)
    Normalize().transform(orientation)
    assert np.allclose(
        orientation.transformed_trajectory,
        np.array([[1 / 3, 2 / 3, 2 / 3], [2 / 3, 2 / 3, 1 / 3]]))


def test_conventional(trajectory):
    orientation = Orientations(trajectory=trajectory,
                               center_type='B',
                               satellite_type='Si',
                               nr_central_atoms=1)
    orientation.transformed_trajectory = np.array([[1, 0, 0], [0, 1, 0]],
                                                  dtype=float)
    Conventional().transform(orientation)
    assert np.allclose(
        orientation.transformed_trajectory,
        np.array([[1 / np.sqrt(2), 1 / np.sqrt(2), 0],
                  [-1 / np.sqrt(6), 1 / np.sqrt(6), 2 / np.sqrt(6)]]))


def test_symmetrize(trajectory):
    orientation = Orientations(trajectory=trajectory,
                               center_type='B',
                               satellite_type='Si',
                               nr_central_atoms=1)
    orientation.transformed_trajectory = np.array([[[1, 0, 0]], [[0, 1, 0]]],
                                                  dtype=float)
    orientation.set_symmetry_operations(
        explicit_sym=np.array([[0, -1, 0], [1, 0, 0], [0, 0, -1]]))
    Symmetrize().transform(orientation)
    assert np.allclose(
        orientation.transformed_trajectory,
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


def test_mean_squared_angular_displacement(trajectory):
    msad = mean_squared_angular_displacement(trajectory.positions)
    assert isinstance(msad, np.ndarray)
    assert isclose(msad.mean(), 0.8142314269325723)
    assert msad.shape == (trajectory.positions.shape[1],
                          trajectory.positions.shape[0])
