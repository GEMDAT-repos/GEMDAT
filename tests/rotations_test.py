from __future__ import annotations

import numpy as np

from gemdat.rotations import Conventional, Normalize, Orientations, Symmetrize


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
