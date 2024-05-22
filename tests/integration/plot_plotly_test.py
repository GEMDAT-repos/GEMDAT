from __future__ import annotations

import numpy as np

from gemdat.io import load_known_material
from gemdat.plots import plotly as plots

import pytest


def test_displacement_per_element(vasp_traj):
    plots.displacement_per_element(trajectory=vasp_traj)


def test_displacement_per_atom(vasp_traj):
    diff_trajectory = vasp_traj.filter('Li')
    plots.displacement_per_atom(trajectory=diff_trajectory)


def test_displacement_histogram(vasp_traj):
    diff_trajectory = vasp_traj.filter('Li')
    plots.displacement_histogram(trajectory=diff_trajectory)


def test_frequency_vs_occurence(vasp_traj):
    diff_traj = vasp_traj.filter('Li')
    plots.frequency_vs_occurence(trajectory=diff_traj)


def test_vibrational_amplitudes(vasp_traj):
    diff_traj = vasp_traj.filter('Li')
    plots.vibrational_amplitudes(trajectory=diff_traj)


def test_jumps_vs_distance(vasp_jumps):
    plots.jumps_vs_distance(jumps=vasp_jumps)


def test_jumps_vs_time(vasp_jumps):
    plots.jumps_vs_time(jumps=vasp_jumps)


def test_collective_jumps(vasp_jumps):
    plots.collective_jumps(jumps=vasp_jumps)


def test_jumps_3d(vasp_jumps):
    plots.jumps_3d(jumps=vasp_jumps)


@pytest.mark.xfail(reason='not implemented yet')
def test_radial_distribution(vasp_rdf_data):
    assert len(vasp_rdf_data) == 3
    for rdfs in vasp_rdf_data.values():
        plots.radial_distribution(rdfs)


@pytest.mark.xfail(reason='not implemented yet')
def test_shape(vasp_shape_data):
    assert len(vasp_shape_data) == 1
    for shape in vasp_shape_data:
        plots.shape(shape)


def test_msd_per_element(vasp_traj):
    plots.msd_per_element(trajectory=vasp_traj[-500:])


@pytest.mark.xfail(reason='not implemented yet')
def test_energy_along_path(vasp_path):
    structure = load_known_material('argyrodite')
    plots.energy_along_path(path=vasp_path, structure=structure)


@pytest.mark.xfail(reason='not implemented yet')
def test_rectilinear(vasp_orientations):
    matrix = np.array(
        [[1 / 2**0.5, -1 / 6**0.5, 1 / 3**0.5],
         [1 / 2**0.5, 1 / 6**0.5, -1 / 3**0.5], [0, 2 / 6**0.5, 1 / 3**0.5]], )

    orientations = vasp_orientations.normalize().transform(matrix=matrix)
    plots.rectilinear(orientations=orientations, normalize_histo=False)


def test_bond_length_distribution(vasp_orientations):
    plots.bond_length_distribution(orientations=vasp_orientations, bins=50)


def test_autocorrelation(vasp_orientations):
    plots.autocorrelation(orientations=vasp_orientations)
