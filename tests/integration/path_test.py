from __future__ import annotations

from math import isclose

import numpy as np
import pytest

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
def test_precompute_nearest_peak(vasp_full_vol):
    peaks = np.array([[30, 30, 30], [31, 30, 30]])
    peaks_map = vasp_full_vol.precompute_nearest_peaks(peaks)

    assert peaks_map.get(tuple((35, 3, 3))) == (31, 30, 30)
    assert peaks_map.get(tuple((64, 3, 3))) == (30, 30, 30)
