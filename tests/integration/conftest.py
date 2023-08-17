from pathlib import Path

import pytest
from gemdat.io import load_known_material
from gemdat.rdf import radial_distribution
from gemdat.sites import SitesData
from gemdat.trajectory import Trajectory

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
