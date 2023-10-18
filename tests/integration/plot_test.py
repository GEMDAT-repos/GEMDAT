from __future__ import annotations

from helpers import image_comparison2

from gemdat import plots


@image_comparison2(baseline_images=['displacement_per_element'])
def test_displacement_per_element(vasp_traj):
    plots.displacement_per_element(trajectory=vasp_traj)


@image_comparison2(baseline_images=['displacement_per_site'])
def test_displacement_per_site(vasp_traj):
    diff_trajectory = vasp_traj.filter('Li')
    plots.displacement_per_site(trajectory=diff_trajectory)


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
def test_jumps_vs_distance(vasp_sites):
    plots.jumps_vs_distance(sites=vasp_sites)


@image_comparison2(baseline_images=['jumps_vs_time'])
def test_jumps_vs_time(vasp_sites):
    plots.jumps_vs_time(sites=vasp_sites)


@image_comparison2(baseline_images=['collective_jumps'])
def test_collective_jumps(vasp_sites):
    plots.collective_jumps(sites=vasp_sites)


@image_comparison2(baseline_images=['jumps_3d'])
def test_jumps_3d(vasp_sites):
    plots.jumps_3d(sites=vasp_sites)


@image_comparison2(baseline_images=['jumps_3d_animation'])
def test_jumps_3d_animation(vasp_sites):
    plots.jumps_3d_animation(sites=vasp_sites, t_start=1000, t_stop=1001)


@image_comparison2(baseline_images=['rdf1', 'rdf2', 'rdf3'])
def test_rdf(vasp_rdf_data):
    for rdfs in vasp_rdf_data.values():
        plots.radial_distribution(rdfs)
