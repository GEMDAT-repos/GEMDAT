from __future__ import annotations

import numpy as np
from helpers import assert_figures_similar

from gemdat.io import load_known_material

BACKEND = 'plotly'


def test_displacement_per_element(vasp_traj):
    fig = vasp_traj.plot_displacement_per_element(backend=BACKEND)

    assert_figures_similar(fig, name='displacement_per_element', rms=0.5)


def test_displacement_per_atom(vasp_traj):
    diff_trajectory = vasp_traj.filter('Li')
    fig = diff_trajectory.plot_displacement_per_atom(backend=BACKEND)

    assert_figures_similar(fig, name='displacement_per_atom', rms=0.5)


def test_displacement_histogram(vasp_traj):
    diff_trajectory = vasp_traj.filter('Li')
    fig = diff_trajectory.plot_displacement_histogram(backend=BACKEND)

    assert_figures_similar(fig, name='displacement_histogram', rms=0.5)


def test_frequency_vs_occurence(vasp_traj):
    diff_traj = vasp_traj.filter('Li')
    fig = diff_traj.plot_frequency_vs_occurence(backend=BACKEND)

    assert_figures_similar(fig, name='frequency_vs_occurence', rms=0.5)


def test_vibrational_amplitudes(vasp_traj):
    diff_traj = vasp_traj.filter('Li')
    fig = diff_traj.plot_vibrational_amplitudes(backend=BACKEND)

    assert_figures_similar(fig, name='vibrational_amplitudes', rms=0.5)


def test_jumps_vs_distance(vasp_jumps):
    fig = vasp_jumps.plot_jumps_vs_distance(backend=BACKEND)

    assert_figures_similar(fig, name='jumps_vs_distance', rms=0.5)


def test_jumps_vs_time(vasp_jumps):
    fig = vasp_jumps.plot_jumps_vs_time(backend=BACKEND)

    assert_figures_similar(fig, name='jumps_vs_time', rms=0.5)


def test_collective_jumps(vasp_jumps):
    fig = vasp_jumps.plot_collective_jumps(backend=BACKEND)

    assert_figures_similar(fig, name='collective_jumps', rms=0.5)


def test_jumps_3d(vasp_jumps):
    fig = vasp_jumps.plot_jumps_3d(backend=BACKEND)

    assert_figures_similar(fig, name='jumps_3d', rms=0.5)


def test_radial_distribution(vasp_rdf_data):
    assert len(vasp_rdf_data) == 3
    for i, rdfs in enumerate(vasp_rdf_data.values()):
        fig = rdfs.plot(backend=BACKEND)

        assert_figures_similar(fig, name=f'radial_distribution_{i}', rms=0.5)


def test_radial_distribution_between_species(vasp_traj):
    from gemdat.rdf import radial_distribution_between_species

    traj = vasp_traj[-500:]
    rdf = radial_distribution_between_species(
        trajectory=traj,
        specie_1='Li',
        specie_2=('S', 'CL'),
    )
    fig = rdf.plot(backend=BACKEND)

    assert_figures_similar(fig, name='radial_distribution_between_species', rms=0.5)


def test_shape(vasp_shape_data):
    assert len(vasp_shape_data) == 1
    for shape in vasp_shape_data:
        fig = shape.plot(backend=BACKEND)

        assert_figures_similar(fig, name='shape', rms=0.5)


def test_msd_per_element(vasp_traj):
    vasp_traj = vasp_traj[-500:]
    fig = vasp_traj.plot_msd_per_element(backend=BACKEND)

    assert_figures_similar(fig, name='msd_per_element', rms=0.5)


def test_energy_along_path(vasp_path):
    structure = load_known_material('argyrodite')
    fig = vasp_path.plot_energy_along_path(backend=BACKEND, structure=structure)

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
    fig = orientations.plot_rectilinear(backend=BACKEND, normalize_histo=False)

    assert_figures_similar(fig, name='rectilinear', rms=0.5)


def test_bond_length_distribution(vasp_orientations):
    fig = vasp_orientations.plot_bond_length_distribution(backend=BACKEND, bins=50)

    assert_figures_similar(fig, name='bond_length_distribution', rms=0.5)


def test_autocorrelation(vasp_orientations):
    fig = vasp_orientations.plot_autocorrelation(backend=BACKEND)

    assert_figures_similar(fig, name='autocorrelation', rms=0.5)
