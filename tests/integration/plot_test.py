from gemdat import plots
from helpers import image_comparison2


@image_comparison2(baseline_images=['displacement_per_element'])
def test_plot_displacement_per_element(vasp_traj):
    plots.plot_displacement_per_element(trajectory=vasp_traj)


@image_comparison2(baseline_images=['displacement_per_site'])
def test_plot_displacement_per_site(vasp_traj):
    diff_trajectory = vasp_traj.filter('Li')
    plots.plot_displacement_per_site(trajectory=diff_trajectory)


@image_comparison2(baseline_images=['displacement_histogram'])
def test_plot_displacement_histogram(vasp_traj):
    diff_trajectory = vasp_traj.filter('Li')
    plots.plot_displacement_histogram(trajectory=diff_trajectory)


@image_comparison2(baseline_images=['frequency_vs_occurence'])
def test_plot_frequency_vs_occurence(vasp_traj):
    plots.plot_frequency_vs_occurence(trajectory=vasp_traj)


@image_comparison2(baseline_images=['vibrational_amplitudes'])
def test_plot_vibrational_amplitudes(vasp_traj):
    plots.plot_vibrational_amplitudes(trajectory=vasp_traj)


# @image_comparison2(baseline_images=['jumps_vs_distance'])
# def test_plot_jumps_vs_distance(vasp_traj):
#     plots.plot_jumps_vs_distance(trajectory=vasp_traj,
#                                  sites=sites)

# @image_comparison2(baseline_images=['jumps_vs_time'])
# def test_plot_jumps_vs_time(vasp_traj):
#     plots.plot_jumps_vs_time(trajectory=vasp_traj, sites=sites)

# @image_comparison2(baseline_images=['collective_jumps'])
# def test_plot_collective_jumps(vasp_traj):
#     plots.plot_collective_jumps(trajectory=vasp_traj, sites=sites)

# @image_comparison2(baseline_images=['jumps_3d'])
# def test_plot_jumps_3d(vasp_traj):
#     plots.plot_jumps_3d(trajectory=vasp_traj, sites=sites)

# @image_comparison2(baseline_images=['jumps_3d_animation'])
# def test_plot_jumps_3d_animation(vasp_traj):
#     plots.plot_jumps_3d_animation(trajectory=vasp_traj, sites=sites)
