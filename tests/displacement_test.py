from types import SimpleNamespace

import numpy as np
from gemdat import SimulationData
from gemdat.calculate import Displacements
from pymatgen.core import Lattice, Species

data = SimulationData(
    trajectory_coords=np.array([[[0, 0, .1]], [[0.5, .45, .9]], [[0, .9, .2]],
                                [[0.5, .42, .8]], [[0, .1, .25]],
                                [[0.5, .59, 0]]]),
    lattice=Lattice(matrix=np.eye(3)),
    species=[Species('Li')],
    time_step=None,
    temperature=None,
    parameters=None,
    structure=None,
)
extras = SimpleNamespace(
    equilibration_steps=1,
    diffusing_element='Li',
)


def test_displacement_calculate_all():
    ret = Displacements.calculate_all(data, extras)
    assert (np.array_equal(
        ret['cell_offsets'],
        np.array([[[0, 0, 0]], [[-1, 0, -1]], [[0, 0, 0]], [[-1, 0, -1]],
                  [[0, 0, 0]], [[-1, 0, 0]]])))
    assert (np.allclose(
        ret['displacements'],
        np.array([[0., 0.73654599, 0.10440307, 0.70356236, 0.17204651]])))
