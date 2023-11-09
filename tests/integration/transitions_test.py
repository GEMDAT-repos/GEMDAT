from __future__ import annotations

from math import isclose

import pytest

from gemdat.path import find_best_perc_path
from gemdat.volume import trajectory_to_volume


@pytest.fixture
def vasp_full_vol(vasp_full_traj):
    trajectory = vasp_full_traj
    diff_trajectory = trajectory.filter('Li')
    return trajectory_to_volume(trajectory=diff_trajectory, resolution=0.3)


@pytest.vaspxml_available  # type: ignore
def test_transitions_find_best_perc_path(vasp_full_vol):
    F = vasp_full_vol.get_free_energy(temperature=650.0)
    peaks = vasp_full_vol.find_peaks()
    path = find_best_perc_path(F,
                               peaks,
                               percolate_x=True,
                               percolate_y=False,
                               percolate_z=False)

    assert isclose(sum(path.energy), 422323.40024684614)
    assert path.sites[0] == (30, 23, 14)
