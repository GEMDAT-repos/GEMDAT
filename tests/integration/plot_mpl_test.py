from __future__ import annotations

import numpy as np
from helpers import image_comparison2

from gemdat.io import load_known_material

BACKEND = 'matplotlib'


@image_comparison2(baseline_images=['displacement_per_element'])
def test_displacement_per_element(vasp_traj):
    vasp_traj.plot_displacement_per_element(backend=BACKEND)


@image_comparison2(baseline_images=['displacement_per_atom'])
def test_displacement_per_atom(vasp_traj):
    diff_trajectory = vasp_traj.filter('Li')
    diff_trajectory.plot_displacement_per_atom(backend=BACKEND)


@image_comparison2(baseline_images=['displacement_histogram'])
def test_displacement_histogram(vasp_traj):
    diff_trajectory = vasp_traj.filter('Li')
    diff_trajectory.plot_displacement_histogram(backend=BACKEND)


@image_comparison2(baseline_images=['frequency_vs_occurence'])
def test_frequency_vs_occurence(vasp_traj):
    diff_traj = vasp_traj.filter('Li')
    diff_traj.plot_frequency_vs_occurence(backend=BACKEND)


@image_comparison2(baseline_images=['vibrational_amplitudes'])
def test_vibrational_amplitudes(vasp_traj):
    diff_traj = vasp_traj.filter('Li')
    diff_traj.plot_vibrational_amplitudes(backend=BACKEND)


@image_comparison2(baseline_images=['jumps_vs_distance'])
def test_jumps_vs_distance(vasp_jumps):
    vasp_jumps.plot_jumps_vs_distance(backend=BACKEND)


@image_comparison2(baseline_images=['jumps_vs_time'])
def test_jumps_vs_time(vasp_jumps):
    vasp_jumps.plot_jumps_vs_time(backend=BACKEND)


@image_comparison2(baseline_images=['collective_jumps'])
def test_collective_jumps(vasp_jumps):
    vasp_jumps.plot_collective_jumps(backend=BACKEND)


@image_comparison2(baseline_images=['jumps_3d'])
def test_jumps_3d(vasp_jumps):
    vasp_jumps.plot_jumps_3d(backend=BACKEND)


@image_comparison2(baseline_images=['jumps_3d_animation'])
def test_jumps_3d_animation(vasp_jumps):
    vasp_jumps.plot_jumps_3d_animation(backend=BACKEND, t_start=1000, t_stop=1001)


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
        rdfs.plot(backend=BACKEND)


@image_comparison2(baseline_images=['radial_distribution_between_species'])
def test_radial_distribution_between_species(vasp_traj):
    from gemdat.rdf import radial_distribution_between_species

    traj = vasp_traj[-500:]
    rdf = radial_distribution_between_species(
        trajectory=traj,
        specie_1='Li',
        specie_2=('S', 'CL'),
    )
    rdf.plot(backend=BACKEND)


@image_comparison2(baseline_images=['shape'])
def test_shape(vasp_shape_data):
    assert len(vasp_shape_data) == 1
    for shape in vasp_shape_data:
        shape.plot(backend=BACKEND)


@image_comparison2(baseline_images=['msd_per_element'])
def test_msd_per_element(vasp_traj):
    traj = vasp_traj[-500:]
    traj.plot_msd_per_element(backend=BACKEND)


@image_comparison2(baseline_images=['energy_along_path'])
def test_energy_along_path(vasp_path):
    structure = load_known_material('argyrodite')
    vasp_path.plot_energy_along_path(backend=BACKEND, structure=structure)


@image_comparison2(baseline_images=['rectilinear'])
def test_rectilinear(vasp_orientations):
    matrix = np.array(
        [
            [1 / 2**0.5, -1 / 6**0.5, 1 / 3**0.5],
            [1 / 2**0.5, 1 / 6**0.5, -1 / 3**0.5],
            [0, 2 / 6**0.5, 1 / 3**0.5],
        ],
    )

    orientations = vasp_orientations.normalize().transform(matrix=matrix)
    orientations.plot_rectilinear(backend=BACKEND, normalize_histo=False)


@image_comparison2(baseline_images=['bond_length_distribution'])
def test_bond_length_distribution(vasp_orientations):
    vasp_orientations.plot_bond_length_distribution(backend=BACKEND, bins=50)


@image_comparison2(baseline_images=['autocorrelation'])
def test_autocorrelation(vasp_orientations):
    vasp_orientations.plot_autocorrelation(backend=BACKEND)
