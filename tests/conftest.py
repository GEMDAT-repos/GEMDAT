from pathlib import Path

import numpy as np
import pytest
from gemdat.io import load_known_material
from gemdat.trajectory import Trajectory
from pymatgen.core import Species

DATA_DIR = Path(__file__).parent / 'data'
VASP_XML = DATA_DIR / 'short_simulation' / 'vasprun.xml'


def pytest_configure():
    pytest.vaspxml_available = pytest.mark.skipif(
        not VASP_XML.exists(),
        reason=
        ('Simulation data from vasprun.xml example is required for this test. '
         'Run `git submodule init`/`update`, and extract using `tar -C tests/data/short_simulation '
         '-xjf tests/data/short_simulation/vasprun.xml.bz2`'))


@pytest.fixture()
def trajectory():
    coords = np.array([
        [[.2, .0, .0], [.0, .0, .5]],
        [[.4, .0, .0], [.0, .0, .5]],
        [[.6, .0, .0], [.0, .0, .5]],
        [[.8, .0, .0], [.0, .0, .5]],
        [[.1, .0, .0], [.0, .0, .5]],
    ])

    return Trajectory(species=[Species('B'), Species('C')],
                      coords=coords,
                      lattice=np.eye(3),
                      metadata={'temperature': 123},
                      time_step=1)


@pytest.fixture(scope='session')
def vasp_traj():
    trajectory = Trajectory.from_vasprun(VASP_XML)
    trajectory = trajectory[1250:]
    return trajectory


@pytest.fixture(scope='session')
def structure():
    return load_known_material('argyrodite', supercell=(2, 1, 1))
