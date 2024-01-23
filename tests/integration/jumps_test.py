from __future__ import annotations

import numpy as np
import pytest

from gemdat import Jumps, Transitions


@pytest.vaspxml_available
class TestSites:  # type: ignore
    n_parts = 10
    diffusing_element = 'Li'

    def test_site_inner_fraction(self, vasp_traj, vasp_sites, structure):
        transitions = Transitions.from_trajectory(trajectory=vasp_traj,
                                                  structure=structure,
                                                  floating_specie='Li',
                                                  site_inner_fraction=0.5)
        jumps = Jumps(transitions=transitions, sites=vasp_sites)

        assert len(jumps.data) == 256
        assert np.all(jumps.data[::100].to_numpy() == np.array([[
            0, 94, 0, 282, 303
        ], [16, 64, 42, 2106, 2179], [35, 77, 15, 2559, 2603]]))

    def test_minimal_residency(self, vasp_traj, vasp_sites, structure):
        transitions = Transitions.from_trajectory(trajectory=vasp_traj,
                                                  structure=structure,
                                                  floating_specie='Li')
        jumps = Jumps(transitions=transitions,
                      sites=vasp_sites,
                      minimal_residence=3)

        assert len(jumps.data) == 457
        assert np.all(jumps.data[::200].to_numpy() == np.array([[
            0, 94, 0, 282, 284
        ], [18, 54, 24, 3336, 3368], [41, 67, 41, 2886, 2893]]))
