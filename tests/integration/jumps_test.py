from __future__ import annotations

import numpy as np
import pytest

from gemdat import Jumps, Transitions


@pytest.vaspxml_available
class TestSites:  # type: ignore
    n_parts = 10
    diffusing_element = 'Li'

    def test_site_inner_fraction(self, vasp_traj, structure):
        transitions = Transitions.from_trajectory(trajectory=vasp_traj,
                                                  structure=structure,
                                                  floating_specie='Li',
                                                  site_inner_fraction=0.5)
        jumps = Jumps(transitions=transitions, minimal_residence=100)

        assert len(jumps.data) == 267
        assert np.all(jumps.data[::100].to_numpy() == np.array([[
            0, 94, 0, 282, 303
        ], [15, 74, 8, 1271, 1286], [34, 49, 33, 3141, 3296]]))

    def test_minimal_residency(self, vasp_traj, structure):
        transitions = Transitions.from_trajectory(trajectory=vasp_traj,
                                                  structure=structure,
                                                  floating_specie='Li')
        jumps = Jumps(transitions=transitions, minimal_residence=3)

        assert len(jumps.data) == 462
        assert np.all(jumps.data[::200].to_numpy() == np.array([[
            0, 94, 0, 282, 284
        ], [18, 54, 24, 2937, 3015], [41, 41, 67, 849, 851]]))
