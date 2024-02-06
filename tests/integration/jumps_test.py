from __future__ import annotations

from math import isclose

import numpy as np
import pandas as pd
import pytest

from gemdat import Jumps, SimulationMetrics, Transitions


@pytest.vaspxml_available
class TestJumps:  # type: ignore
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

    def test_n_jumps(self, vasp_jumps):
        assert vasp_jumps.n_solo_jumps == 450
        assert vasp_jumps.n_jumps == 462
        assert isclose(vasp_jumps.solo_fraction, 0.974026, abs_tol=1e-4)

    def test_rates(self, vasp_jumps):
        rates = vasp_jumps.rates(n_parts=10)
        assert isinstance(rates, pd.DataFrame)
        assert len(rates) == 1

        row = rates.loc[('48h', '48h')]

        assert isclose(row['rates'], 1174999999999.9998)
        assert isclose(row['std'], 90769080986.31458)

    def test_activation_energies(self, vasp_jumps):
        activation_energies = vasp_jumps.activation_energies(n_parts=10)

        assert isinstance(activation_energies, pd.DataFrame)
        assert len(activation_energies) == 1

        row = activation_energies.loc[('48h', '48h')]

        assert isclose(row['energy'], 0.134486, abs_tol=1e-4)
        assert isclose(row['std'], 0.00405952, abs_tol=1e-6)

    def test_jump_diffusivity(self, vasp_jumps):
        assert isclose(vasp_jumps.jump_diffusivity(3),
                       9.484382e-09,
                       rel_tol=1e-6)

    def test_correlation_factor(self, vasp_traj, vasp_jumps):
        vasp_diff_traj = vasp_traj.filter('Li')
        metrics = SimulationMetrics(vasp_diff_traj)
        tracer_diff = metrics.tracer_diffusivity(dimensions=3)
        correlation_factor = tracer_diff / vasp_jumps.jump_diffusivity(
            dimensions=3)
        assert isclose(correlation_factor, 0.165600, rel_tol=1e-6)

    def test_collective(self, vasp_jumps):
        collective = vasp_jumps.collective()
        cc = collective.collective

        assert len(cc) == 6
        assert cc[3][0]['start site'] == 60
        assert cc[3][1]['start site'] == 68

    def test_coll_jumps(self, vasp_jumps):
        collective = vasp_jumps.collective()
        coll_jumps = collective.coll_jumps
        assert len(coll_jumps) == 6
        assert vasp_jumps.n_jumps == collective.n_solo_jumps + collective.n_coll_jumps
        assert coll_jumps == [((49, 31), (33, 49)), ((54, 38), (24, 54)),
                              ((42, 75), (64, 42)), ((60, 36), (68, 60)),
                              ((36, 52), (80, 36)), ((2, 58), (92, 2))]

    def test_collective_matrix(self, vasp_jumps):
        collective = vasp_jumps.collective()
        matrix = collective.site_pair_count_matrix()
        assert matrix.shape == (1, 1)
        assert matrix[0, 0] == 6

    def test_multiple_collective(self, vasp_jumps):
        collective = vasp_jumps.collective()
        jumps, counts = collective.multiple_collective()
        assert jumps.shape == (6, 2)
        assert counts.sum() == 6
        assert np.all(
            jumps == np.array([[(2, 58), (92, 2)], [(24, 54), (
                54, 38)], [(33, 49), (49, 31)], [(36, 52), (
                    80, 36)], [(42, 75), (64, 42)], [(60, 36), (68, 60)]],
                              dtype=[('start', '<i8'), ('stop', '<i8')]))
