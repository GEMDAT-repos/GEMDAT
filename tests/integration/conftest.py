from __future__ import annotations

from pathlib import Path

import pytest

from gemdat.io import load_known_material
from gemdat.jumps import Jumps
from gemdat.orientations import Orientations
from gemdat.rdf import radial_distribution
from gemdat.shape import ShapeAnalyzer
from gemdat.trajectory import Trajectory
from gemdat.volume import trajectory_to_volume

DATA_DIR = Path(__file__).parents[1] / 'data'
VASP_XML = DATA_DIR / 'short_simulation' / 'vasprun.xml'
VASP_ORI_CACHE = DATA_DIR / 'short_simulation' / 'vasprun_rotations.cache'


def pytest_configure():
    pytest.vaspxml_available = pytest.mark.skipif(
        not VASP_XML.exists(),
        reason=(
            'Simulation data from vasprun.xml example is required for this test. '
            'Run `git submodule init`/`update`, and extract using '
            '`tar -C tests/data/short_simulation '
            '-xjf tests/data/short_simulation/vasprun.xml.bz2`'
        ),
    )
    pytest.vasporicache_available = pytest.mark.skipif(
        not VASP_ORI_CACHE.exists(),
        reason=(
            'Data from vasprun_rotations.cache example is needed for this test. '
            'Run `git submodule init`/`update`, and extract using '
            '`tar -C tests/data/short_simulation '
            '-xjf tests/data/short_simulation/vasprun.xml.bz2`'
        ),
    )


@pytest.fixture(scope='module')
def vasp_traj():
    trajectory = Trajectory.from_vasprun(VASP_XML)
    trajectory = trajectory[1250:]
    return trajectory


@pytest.fixture(scope='module')
def vasp_traj_orientations():
    trajectory = Trajectory.from_cache(VASP_ORI_CACHE)
    return trajectory


@pytest.fixture(scope='module')
def vasp_full_traj():
    trajectory = Trajectory.from_vasprun(VASP_XML)
    return trajectory


@pytest.fixture(scope='module')
def structure():
    return load_known_material('argyrodite', supercell=(2, 1, 1))


@pytest.fixture(scope='module')
def vasp_transitions(vasp_traj, structure):
    transitions = vasp_traj.transitions_between_sites(
        sites=structure, floating_specie='Li'
    )
    return transitions


@pytest.fixture(scope='module')
def vasp_jumps(vasp_transitions):
    return Jumps(transitions=vasp_transitions)


@pytest.fixture(scope='module')
def vasp_rdf_data(vasp_traj, structure):
    # Shorten trajectory for faster test
    trajectory = vasp_traj[-1000:]
    transitions = trajectory.transitions_between_sites(structure, floating_specie='Li')

    rdfs = radial_distribution(
        transitions=transitions,
        floating_specie='Li',
        max_dist=5,
    )

    return rdfs


@pytest.fixture(scope='module')
def vasp_shape_data(vasp_traj):
    trajectory = vasp_traj[-250:]
    trajectory.filter('Li')

    # shape analysis needs structure without supercell
    sites = load_known_material('argyrodite')

    sa = ShapeAnalyzer.from_structure(sites)

    shapes = sa.analyze_trajectory(trajectory, supercell=(2, 1, 1))

    return shapes


@pytest.fixture(scope='module')
def vasp_path_vol(vasp_full_traj):
    trajectory = vasp_full_traj
    diff_trajectory = trajectory.filter('Li')
    return trajectory_to_volume(trajectory=diff_trajectory, resolution=0.7)


@pytest.fixture(scope='module')
def vasp_path(vasp_path_vol):
    peaks = vasp_path_vol.find_peaks()
    free_energy = vasp_path_vol.get_free_energy(temperature=650.0)
    path = free_energy.optimal_percolating_path(peaks=peaks, percolate='x')
    return path


@pytest.fixture(scope='module')
def vasp_F_graph(vasp_path_vol):
    free_energy = vasp_path_vol.get_free_energy(temperature=650.0)
    return free_energy.free_energy_graph()


@pytest.fixture(scope='module')
def vasp_orientations(vasp_traj_orientations):
    central_atoms = 'S'
    satellite_atoms = 'O'
    orientations = Orientations(vasp_traj_orientations, central_atoms, satellite_atoms)

    return orientations


@pytest.fixture(scope='module')
def vasp_orientations_spherical(vasp_orientations):
    cf = vasp_orientations.get_conventional_coordinates()
    cf_spheric = cf.get_vectors_spherical()
    return cf_spheric
