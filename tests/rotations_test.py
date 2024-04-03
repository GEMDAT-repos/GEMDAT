from __future__ import annotations

from math import isclose

import numpy as np
from pymatgen.core import Species

from gemdat.rotations import calculate_spherical_areas, mean_squared_angular_displacement


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
