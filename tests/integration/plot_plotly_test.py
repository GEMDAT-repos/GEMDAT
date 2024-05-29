from __future__ import annotations

from helpers import assert_figures_similar
import numpy as np

from gemdat.io import load_known_material
from gemdat.plots import plotly as plots
import pytest


def test_displacement_per_element(vasp_traj):
    fig = plots.displacement_per_element(trajectory=vasp_traj)

    assert_figures_similar(fig, name='displacement_per_element', rms=0.5)


def test_displacement_per_atom(vasp_traj):
    diff_trajectory = vasp_traj.filter('Li')
    fig = plots.displacement_per_atom(trajectory=diff_trajectory)

    assert_figures_similar(fig, name='displacement_per_atom', rms=0.5)


def test_displacement_histogram(vasp_traj):
    diff_trajectory = vasp_traj.filter('Li')
    fig = plots.displacement_histogram(trajectory=diff_trajectory)

    assert_figures_similar(fig, name='displacement_histogram', rms=0.5)


def test_frequency_vs_occurence(vasp_traj):
    diff_traj = vasp_traj.filter('Li')
    fig = plots.frequency_vs_occurence(trajectory=diff_traj)

    assert_figures_similar(fig, name='frequency_vs_occurence', rms=0.5)


def test_vibrational_amplitudes(vasp_traj):
    diff_traj = vasp_traj.filter('Li')
    fig = plots.vibrational_amplitudes(trajectory=diff_traj)

    assert_figures_similar(fig, name='vibrational_amplitudes', rms=0.5)


def test_jumps_vs_distance(vasp_jumps):
    fig = plots.jumps_vs_distance(jumps=vasp_jumps)

    assert_figures_similar(fig, name='jumps_vs_distance', rms=0.5)


def test_jumps_vs_time(vasp_jumps):
    fig = plots.jumps_vs_time(jumps=vasp_jumps)

    assert_figures_similar(fig, name='jumps_vs_time', rms=0.5)


def test_collective_jumps(vasp_jumps):
    fig = plots.collective_jumps(jumps=vasp_jumps)

    assert_figures_similar(fig, name='collective_jumps', rms=0.5)


def test_jumps_3d(vasp_jumps):
    fig = plots.jumps_3d(jumps=vasp_jumps)

    assert_figures_similar(fig, name='jumps_3d', rms=0.5)


def test_radial_distribution(vasp_rdf_data):
    assert len(vasp_rdf_data) == 3
    for rdfs in vasp_rdf_data.values():
        fig = plots.radial_distribution(rdfs)

    assert_figures_similar(fig, name='radial_distribution', rms=0.5)


@pytest.mark.xfail(reason='not implemented yet')
def test_shape(vasp_shape_data):
    assert len(vasp_shape_data) == 1
    for i, shape in vasp_shape_data:
        fig = plots.shape(shape)

        assert_figures_similar(fig, name='shape_{i}', rms=0.5)


def test_msd_per_element(vasp_traj):
    fig = plots.msd_per_element(trajectory=vasp_traj[-500:])

    assert_figures_similar(fig, name='msd_per_element', rms=0.5)


def test_energy_along_path(vasp_path):
    structure = load_known_material('argyrodite')
    fig = plots.energy_along_path(path=vasp_path, structure=structure)

    assert_figures_similar(fig, name='energy_along_path', rms=0.5)


def test_rectilinear(vasp_orientations):
    matrix = np.array(
        [
            [1 / 2**0.5, -1 / 6**0.5, 1 / 3**0.5],
            [1 / 2**0.5, 1 / 6**0.5, -1 / 3**0.5],
            [0, 2 / 6**0.5, 1 / 3**0.5],
        ],
    )

    orientations = vasp_orientations.normalize().transform(matrix=matrix)
    fig = plots.rectilinear(orientations=orientations, normalize_histo=False)

    assert_figures_similar(fig, name='rectilinear', rms=0.5)


def test_bond_length_distribution(vasp_orientations):
    fig = plots.bond_length_distribution(orientations=vasp_orientations, bins=50)

    assert_figures_similar(fig, name='bond_length_distribution', rms=0.5)


def test_autocorrelation(vasp_orientations):
    fig = plots.autocorrelation(orientations=vasp_orientations)

    assert_figures_similar(fig, name='autocorrelation', rms=0.5)
