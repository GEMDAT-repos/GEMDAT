from __future__ import annotations

import numpy as np
from helpers import image_comparison2

from gemdat.io import load_known_material
from gemdat.plots import matplotlib as plots


@image_comparison2(baseline_images=['displacement_per_element'])
def test_displacement_per_element(vasp_traj):
    plots.displacement_per_element(trajectory=vasp_traj)


@image_comparison2(baseline_images=['displacement_per_atom'])
def test_displacement_per_atom(vasp_traj):
    diff_trajectory = vasp_traj.filter('Li')
    plots.displacement_per_atom(trajectory=diff_trajectory)


@image_comparison2(baseline_images=['displacement_histogram'])
def test_displacement_histogram(vasp_traj):
    diff_trajectory = vasp_traj.filter('Li')
    plots.displacement_histogram(trajectory=diff_trajectory)


@image_comparison2(baseline_images=['frequency_vs_occurence'])
def test_frequency_vs_occurence(vasp_traj):
    diff_traj = vasp_traj.filter('Li')
    plots.frequency_vs_occurence(trajectory=diff_traj)


@image_comparison2(baseline_images=['vibrational_amplitudes'])
def test_vibrational_amplitudes(vasp_traj):
    diff_traj = vasp_traj.filter('Li')
    plots.vibrational_amplitudes(trajectory=diff_traj)


@image_comparison2(baseline_images=['jumps_vs_distance'])
def test_jumps_vs_distance(vasp_jumps):
    plots.jumps_vs_distance(jumps=vasp_jumps)


@image_comparison2(baseline_images=['jumps_vs_time'])
def test_jumps_vs_time(vasp_jumps):
    plots.jumps_vs_time(jumps=vasp_jumps)


@image_comparison2(baseline_images=['collective_jumps'])
def test_collective_jumps(vasp_jumps):
    plots.collective_jumps(jumps=vasp_jumps)


@image_comparison2(baseline_images=['jumps_3d'])
def test_jumps_3d(vasp_jumps):
    plots.jumps_3d(jumps=vasp_jumps)


@image_comparison2(baseline_images=['jumps_3d_animation'])
def test_jumps_3d_animation(vasp_jumps):
    plots.jumps_3d_animation(jumps=vasp_jumps, t_start=1000, t_stop=1001)


@image_comparison2(
    baseline_images=[
        'radial_distribution_1',
        'radial_distribution_2',
        'radial_distribution_3',
    ]
)
def test_radial_distribution(vasp_rdf_data):
    assert len(vasp_rdf_data) == 3
    for rdfs in vasp_rdf_data.values():
        plots.radial_distribution(rdfs)


@image_comparison2(baseline_images=['shape'])
def test_shape(vasp_shape_data):
    assert len(vasp_shape_data) == 1
    for shape in vasp_shape_data:
        plots.shape(shape)


@image_comparison2(baseline_images=['msd_per_element'])
def test_msd_per_element(vasp_traj):
    plots.msd_per_element(trajectory=vasp_traj[-500:])


@image_comparison2(baseline_images=['energy_along_path'])
def test_energy_along_path(vasp_path):
    structure = load_known_material('argyrodite')
    plots.energy_along_path(path=vasp_path, structure=structure)


@image_comparison2(baseline_images=['rectilinear', 'rectilinear_wpeaks'])
def test_rectilinear(vasp_orientations):
    matrix = np.array(
        [
            [1 / 2**0.5, -1 / 6**0.5, 1 / 3**0.5],
            [1 / 2**0.5, 1 / 6**0.5, -1 / 3**0.5],
            [0, 2 / 6**0.5, 1 / 3**0.5],
        ],
    )

    orientations = vasp_orientations.normalize().transform(matrix=matrix)
    plots.rectilinear(orientations=orientations, normalize_histo=False)
    plots.rectilinear(orientations=orientations, normalize_histo=False, add_peaks=True)


@image_comparison2(baseline_images=['bond_length_distribution'])
def test_bond_length_distribution(vasp_orientations):
    plots.bond_length_distribution(orientations=vasp_orientations, bins=50)


@image_comparison2(baseline_images=['autocorrelation'])
def test_autocorrelation(vasp_orientations):
    plots.autocorrelation(orientations=vasp_orientations)
