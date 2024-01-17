from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gemdat.io import load_known_material
from gemdat.path import find_best_perc_path, free_energy_graph
from gemdat.rdf import radial_distribution
from gemdat.shape import ShapeAnalyzer
from gemdat.sites import SitesData
from gemdat.trajectory import Trajectory
from gemdat.volume import trajectory_to_volume

DATA_DIR = Path(__file__).parents[1] / 'data'
VASP_XML = DATA_DIR / 'short_simulation' / 'vasprun.xml'


def pytest_configure():
    pytest.vaspxml_available = pytest.mark.skipif(
        not VASP_XML.exists(),
        reason=
        ('Simulation data from vasprun.xml example is required for this test. '
         'Run `git submodule init`/`update`, and extract using `tar -C tests/data/short_simulation '
         '-xjf tests/data/short_simulation/vasprun.xml.bz2`'))


@pytest.fixture(scope='module')
def vasp_traj():
    trajectory = Trajectory.from_vasprun(VASP_XML)
    trajectory = trajectory[1250:]
    return trajectory


@pytest.fixture(scope='module')
def vasp_full_traj():
    trajectory = Trajectory.from_vasprun(VASP_XML)
    return trajectory


@pytest.fixture(scope='module')
def structure():
    return load_known_material('argyrodite', supercell=(2, 1, 1))


@pytest.fixture(scope='module')
def vasp_sites(vasp_traj, structure):
    sites = SitesData(structure=structure,
                      trajectory=vasp_traj,
                      floating_specie='Li')
    return sites


@pytest.fixture(scope='module')
def vasp_rdf_data(vasp_traj, structure):
    # Shorten trajectory for faster test
    trajectory = vasp_traj[-1000:]

    sites = SitesData(structure=structure,
                      trajectory=trajectory,
                      floating_specie='Li')

    rdfs = radial_distribution(
        sites=sites,
        max_dist=5,
    )

    return rdfs


@pytest.fixture(scope='module')
def vasp_shape_data(vasp_traj):
    trajectory = vasp_traj[-1000:]
    trajectory.filter('Li')

    # shape analysis needs structure without supercell
    structure = load_known_material('argyrodite')

    sa = ShapeAnalyzer.from_structure(structure)

    shapes = sa.analyze_trajectory(trajectory, supercell=(2, 1, 1))

    return shapes


@pytest.fixture(scope='module')
def vasp_full_vol(vasp_full_traj):
    trajectory = vasp_full_traj
    diff_trajectory = trajectory.filter('Li')
    return trajectory_to_volume(trajectory=diff_trajectory, resolution=0.3)


@pytest.fixture(scope='module')
def vasp_full_path(vasp_full_vol):
    F = vasp_full_vol.get_free_energy(temperature=650.0)
    peaks = np.array([[30, 23, 14], [35, 2, 7]])
    path = find_best_perc_path(F,
                               peaks,
                               percolate_x=True,
                               percolate_y=False,
                               percolate_z=False)
    return path


@pytest.fixture(scope='module')
def vasp_F_graph(vasp_full_vol):
    F = vasp_full_vol.get_free_energy(temperature=650.0)
    F_graph = free_energy_graph(F, max_energy_threshold=1e7, diagonal=True)

    return F_graph
