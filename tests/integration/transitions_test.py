from __future__ import annotations

from math import isclose

import pytest

from gemdat.transitions import find_best_perc_path
from gemdat.volume import trajectory_to_volume


@pytest.fixture
def vasp_full_vol(vasp_full_traj):
    trajectory = vasp_full_traj
    diff_trajectory = trajectory.filter('Li')
    return trajectory_to_volume(trajectory=diff_trajectory, resolution=0.3)


@pytest.vaspxml_available  # type: ignore
def test_transitions_find_best_perc_path(vasp_full_vol):
    F = vasp_full_vol.get_free_energy(kBT=650.0)
    peaks = vasp_full_vol.find_peaks()
    total_energy_cost, starting_point, best_perc_path, best_perc_path_energy = find_best_perc_path(
        F, peaks, percolate_x=True, percolate_y=False, percolate_z=False)

    assert isclose(total_energy_cost, 422323.40024684614)
    assert (starting_point == [30, 23, 14]).all()
