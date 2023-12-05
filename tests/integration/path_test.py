from __future__ import annotations

from math import isclose

import numpy as np
import pytest

from gemdat.io import load_known_material
from gemdat.path import find_best_perc_path
from gemdat.volume import trajectory_to_volume


@pytest.fixture
def vasp_full_vol(vasp_full_traj):
    trajectory = vasp_full_traj
    diff_trajectory = trajectory.filter('Li')
    return trajectory_to_volume(trajectory=diff_trajectory, resolution=0.3)


@pytest.vaspxml_available  # type: ignore
def test_find_best_perc_path(vasp_full_vol):
    F = vasp_full_vol.get_free_energy(temperature=650.0)
    peaks = np.array([[30, 23, 14], [35, 2, 7]])

    path = find_best_perc_path(F,
                               peaks,
                               percolate_x=True,
                               percolate_y=False,
                               percolate_z=False)

    assert isclose(path.cost, 36.39301483423)
    assert path.start_site == (30, 23, 14)


@pytest.vaspxml_available  # type: ignore
def test_nearest_structure_reference(vasp_full_vol):
    structure = load_known_material('argyrodite')
    nearest_structure_map = vasp_full_vol.nearest_structure_reference(
        structure)

    assert nearest_structure_map[(0, 0, 0)] == 21
    assert nearest_structure_map[(14.7, 0.8999999999999999,
                                  0.8999999999999999)] == 9
    assert nearest_structure_map[(18.3, 0.8999999999999999, 9.6)] == 37
    assert (18.3, 0.899999999999999, 9.6) not in nearest_structure_map
    assert len(nearest_structure_map) == 71874
