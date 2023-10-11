import numpy as np
import pytest
from gemdat.trajectory import Trajectory
from pymatgen.core import Species


def pytest_addoption(parser):
    """Add options to pytest."""
    parser.addoption('--dashboard',
                     action='store_true',
                     default=False,
                     help='Run dashboard workflow tests')


def pytest_configure(config):
    pytest.skip_dashboard = not config.getoption('--dashboard')


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
