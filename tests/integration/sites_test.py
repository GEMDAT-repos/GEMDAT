"""
Run integration test with:

VASP_XML=/home/stef/md-analysis-matlab-example/vasprun.xml pytest
"""
from __future__ import annotations

from math import isclose

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from gemdat import SimulationMetrics


@pytest.vaspxml_available
class TestSites:  # type: ignore
    n_parts = 10
    diffusing_element = 'Li'

    def test_atom_sites(self, vasp_traj, vasp_transitions):
        n_steps = len(vasp_traj)
        n_diffusing = 48

        assert isclose(vasp_transitions.dist_close, 0.9284961123176741)

        slice_ = np.s_[::1000, ::24]

        states = vasp_transitions.states
        assert states.shape == (n_steps, n_diffusing)
        assert states.sum() == 6154859
        assert_allclose(states[slice_],
                        np.array([[94, -1], [94, -1], [0, 65], [0, 65]]))

        states_next = vasp_transitions.states_next()
        assert states_next.shape == (n_steps, n_diffusing)
        assert states_next.sum() == 8172006
        assert_allclose(states_next[slice_],
                        np.array([[94, 1], [94, 65], [0, 65], [0, 65]]))

        states_prev = vasp_transitions.states_prev()
        assert states_prev.shape == (n_steps, n_diffusing)
        assert states_prev.sum() == 8148552
        assert_allclose(states_prev[slice_],
                        np.array([[94, -1], [94, 65], [0, 65], [0, 65]]))

    def test_n_floating(self, vasp_transitions):
        # https://github.com/GEMDAT-repos/GEMDAT/issues/252
        assert vasp_transitions.n_floating == 48

    def test_n_states(self, vasp_transitions):
        assert vasp_transitions.n_states == 3750

    def test_all_transitions_events(self, vasp_transitions):
        events = vasp_transitions.events

        assert isinstance(events, pd.DataFrame)
        assert vasp_transitions.n_events == 2105
        assert vasp_transitions.events.shape == (2105, 6)
        assert_allclose(
            events[::1000],
            np.array([[0, 94, -1, 94, -1, 202], [22, 70, -1, 70, -1, 3472],
                      [45, 10, -1, 10, -1, 2386]]))

    def test_all_transitions_matrix(self, vasp_transitions):
        matrix = vasp_transitions.matrix()
        n_sites = vasp_transitions.n_sites
        assert matrix.shape == (n_sites, n_sites)
        assert matrix.sum() == 2099
        assert_allclose(
            np.argwhere(matrix)[::50],
            np.array([[0, 95], [49, 95], [95, 5], [95, 55]]))

    def test_transitions_parts(self, vasp_transitions):
        n_sites = vasp_transitions.n_sites
        tp = np.stack(
            [part.matrix() for part in vasp_transitions.split(n_parts=10)])

        assert tp.shape == (self.n_parts, n_sites, n_sites)
        assert tp.sum() == 2100
        assert_allclose(
            np.argwhere(tp)[::500],
            np.array([[0, 0, 95], [4, 23, 95], [8, 28, 95]]))

    def test_occupancy(self, vasp_transitions):
        structure = vasp_transitions.occupancy()

        assert len(structure) == 96
        assert structure[0].species.num_atoms == 0.4544
        assert structure.composition.num_atoms == 36.54026666666665

    def test_occupancy_parts(self, vasp_transitions):
        parts = vasp_transitions.split(5)
        structures = [part.occupancy() for part in parts]

        values1 = [structure[0].species.num_atoms for structure in structures]
        assert values1 == [
            0.11066666666666666, 0.708, 0.408, 0.62, 0.42533333333333334
        ]

        values2 = [structure.composition.num_atoms for structure in structures]
        assert values2 == [
            36.653333333333336, 37.129333333333335, 37.98933333333333,
            35.49199999999999, 35.43733333333334
        ]

    def test_atom_locations(self, vasp_transitions):
        dct = vasp_transitions.atom_locations()
        assert dct == {'48h': 0.7612555555555552}

    def test_atom_locations_parts(self, vasp_transitions):
        parts = vasp_transitions.split(5)
        dcts = [part.atom_locations() for part in parts]

        assert dcts == [
            {
                '48h': 0.7636111111111111
            },
            {
                '48h': 0.7735277777777778
            },
            {
                '48h': 0.7914444444444443
            },
            {
                '48h': 0.7394166666666665
            },
            {
                '48h': 0.7382777777777779
            },
        ]

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
