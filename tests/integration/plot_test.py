from gemdat import plots
from helpers import image_comparison2


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
    plots.frequency_vs_occurence(trajectory=vasp_traj)


@image_comparison2(baseline_images=['vibrational_amplitudes'])
def test_vibrational_amplitudes(vasp_traj):
    plots.vibrational_amplitudes(trajectory=vasp_traj)


@image_comparison2(baseline_images=['jumps_vs_distance'])
def test_jumps_vs_distance(vasp_traj, vasp_sites):
    plots.jumps_vs_distance(trajectory=vasp_traj, sites=vasp_sites)


@image_comparison2(baseline_images=['jumps_vs_time'])
def test_jumps_vs_time(vasp_traj, vasp_sites):
    plots.jumps_vs_time(trajectory=vasp_traj, sites=vasp_sites)


@image_comparison2(baseline_images=['collective_jumps'])
def test_collective_jumps(vasp_traj, vasp_sites):
    plots.collective_jumps(trajectory=vasp_traj, sites=vasp_sites)


@image_comparison2(baseline_images=['jumps_3d'])
def test_jumps_3d(vasp_traj, vasp_sites):
    plots.jumps_3d(trajectory=vasp_traj, sites=vasp_sites)


@image_comparison2(baseline_images=['jumps_3d_animation'])
def test_jumps_3d_animation(vasp_traj, vasp_sites):
    plots.jumps_3d_animation(trajectory=vasp_traj,
                             sites=vasp_sites,
                             t_start=1000,
                             t_stop=1001)


@image_comparison2(baseline_images=['rdf1', 'rdf2', 'rdf3'])
def test_rdf(vasp_traj, structure):
    from gemdat.rdf import radial_distribution
    from gemdat.sites import SitesData

    trajectory = vasp_traj[-1000:]

    sites = SitesData(structure=structure,
                      trajectory=trajectory,
                      floating_specie='Li')

    rdfs = radial_distribution(
        trajectory=trajectory,
        sites=sites,
        species='Li',
        max_dist=5,
    )
    for state, rdf in rdfs.items():
        plots.radial_distribution(rdf, name=state)
