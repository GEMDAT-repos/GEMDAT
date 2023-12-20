from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Species

from gemdat.trajectory import Trajectory


@pytest.fixture()
def trajectory():
    coords = np.array([
        [[.2, .0, .0], [.0, .0, .5], [.0, .0, .5], [.0, .0, .5]],
        [[.4, .0, .0], [.0, .0, .5], [.0, .0, .5], [.0, .0, .5]],
        [[.6, .0, .0], [.0, .0, .5], [.0, .0, .5], [.0, .0, .5]],
        [[.8, .0, .0], [.0, .0, .5], [.0, .0, .5], [.0, .0, .5]],
        [[.1, .0, .0], [.0, .0, .5], [.0, .0, .5], [.0, .0, .5]],
    ])

    return Trajectory(
        species=[Species('B'),
                 Species('Si'),
                 Species('S'),
                 Species('C')],
        coords=coords,
        lattice=np.eye(3),
        metadata={'temperature': 123},
        time_step=1)
