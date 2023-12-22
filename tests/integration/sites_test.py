"""
Run integration test with:

VASP_XML=/home/stef/md-analysis-matlab-example/vasprun.xml pytest
"""
from __future__ import annotations

from math import isclose

import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.vaspxml_available
class TestSites:  # type: ignore
    n_parts = 10
    diffusing_element = 'Li'

    def test_atom_sites(self, vasp_sites, vasp_traj):
        n_steps = len(vasp_traj)
        n_diffusing = vasp_sites.n_floating

        transitions = vasp_sites.transitions

        assert isclose(transitions._dist_close, 0.9284961123176741)

        slice_ = np.s_[::1000, ::24]

        states = transitions.states
        assert states.shape == (n_steps, n_diffusing)
        assert states.sum() == 6154859
        assert_allclose(states[slice_],
                        np.array([[94, -1], [94, -1], [0, 65], [0, 65]]))

        states_next = transitions.states_next()
        assert states_next.shape == (n_steps, n_diffusing)
        assert states_next.sum() == 8172006
        assert_allclose(states_next[slice_],
                        np.array([[94, 1], [94, 65], [0, 65], [0, 65]]))

        states_prev = transitions.states_prev()
        assert states_prev.shape == (n_steps, n_diffusing)
        assert states_prev.sum() == 8148552
        assert_allclose(states_prev[slice_],
                        np.array([[94, -1], [94, 65], [0, 65], [0, 65]]))

    def test_all_transitions(self, vasp_sites):
        transitions = vasp_sites.transitions

        events = transitions.events
        assert transitions.events.shape == (2105, 4)
        assert_allclose(
            events[::1000],
            np.array([[0, 94, -1, 202], [22, 70, -1, 3472], [45, 10, -1,
                                                             2386]]))

        matrix = transitions.matrix()
        assert matrix.shape == (vasp_sites.n_sites, vasp_sites.n_sites)
        assert matrix.sum() == 2099
        assert_allclose(
            np.argwhere(matrix)[::50],
            np.array([[0, 95], [49, 95], [95, 5], [95, 55]]))

    def test_transitions_parts(self, vasp_sites):
        n_sites = vasp_sites.n_sites
        tp = vasp_sites.transitions_parts

        assert tp.shape == (self.n_parts, n_sites, n_sites)
        assert tp.sum() == 2100
        assert_allclose(
            np.argwhere(tp)[::500],
            np.array([[0, 0, 95], [4, 23, 95], [8, 28, 95]]))

    def test_occupancy(self, vasp_sites):
        occupancy = vasp_sites.transitions.occupancy()
        assert len(occupancy) == 95
        assert sum(occupancy.values()) == 137026
        assert list(occupancy.values())[::20] == [1704, 971, 351, 1508, 1104]

    def test_occupancy_parts(self, vasp_sites):
        parts = vasp_sites.occupancy_parts
        assert len(parts) == self.n_parts
        assert [sum(part.values()) for part in parts] == [
            13592, 13898, 13819, 14028, 14022, 14470, 13200, 13419, 13286,
            13292
        ]

    def test_site_occupancy(self, vasp_sites):
        assert isclose(vasp_sites.site_occupancy()['48h'],
                       0.380628,
                       rel_tol=1e-4)

    def test_site_occupancy_parts(self, vasp_sites):
        assert len(vasp_sites.site_occupancy_parts) == self.n_parts
        assert isclose(vasp_sites.site_occupancy_parts[0]['48h'],
                       0.37756,
                       rel_tol=1e-4)
        assert isclose(vasp_sites.site_occupancy_parts[9]['48h'],
                       0.36922,
                       rel_tol=1e-4)

    def test_atom_locations(self, vasp_sites):
        assert isclose(vasp_sites.atom_locations()['48h'],
                       0.761255,
                       rel_tol=1e-4)

    def test_atom_locations_parts(self, vasp_sites):
        assert len(vasp_sites.atom_locations_parts()) == self.n_parts
        assert isclose(vasp_sites.atom_locations_parts()[0]['48h'],
                       0.755111,
                       rel_tol=1e-4)
        assert isclose(vasp_sites.atom_locations_parts()[9]['48h'],
                       0.738444,
                       rel_tol=1e-4)

    def test_n_jumps(self, vasp_jumps):
        assert vasp_jumps.n_solo_jumps == 1
        assert vasp_jumps.n_jumps == 462
        assert isclose(vasp_jumps.solo_fraction, 0.00216, abs_tol=1e-4)

    def test_rates(self, vasp_jumps):
        rates = vasp_jumps.rates()
        assert isinstance(rates, dict)
        assert len(rates) == 1

        rates, rates_std = rates[('48h', '48h')]
        assert isclose(rates, 1174999999999.9998)
        assert isclose(rates_std, 90769080986.31458)

    def test_activation_energies(self, vasp_jumps, vasp_sites):
        activation_energies = vasp_jumps.activation_energies()

        assert isinstance(activation_energies, dict)
        assert len(activation_energies) == 1

        e_act, e_act_std = activation_energies[('48h', '48h')]

        assert isclose(e_act, 0.134486260, abs_tol=1e-4)
        assert isclose(e_act_std, .00405951, abs_tol=1e-6)

    def test_jump_diffusivity(self, vasp_jumps):
        assert isclose(vasp_jumps.jump_diffusivity(3),
                       9.484382424533019e-09,
                       rel_tol=1e-6)

    def test_correlation_factor(self, vasp_sites, vasp_jumps):
        tracer_diff = vasp_sites.metrics.tracer_diffusivity(dimensions=3)
        correlation_factor = tracer_diff / vasp_jumps.jump_diffusivity(
            dimensions=3)
        assert isclose(correlation_factor, 0.1656001328253986, rel_tol=1e-6)

    def test_collective(self, vasp_jumps):
        collective = vasp_jumps.collective()
        cc = collective.collective

        assert len(cc) == 1587
        assert cc[1000][0]['start site'] == 23
        assert cc[1000][1]['start site'] == 95

    def test_coll_jumps(self, vasp_jumps):
        collective = vasp_jumps.collective()
        coll_jumps = collective.coll_jumps

        assert len(coll_jumps) == 1587
        assert coll_jumps[::1000] == [((74, 8), (41, 67)),
                                      ((23, 31), (95, 79))]

    def test_collective_matrix(self, vasp_jumps):
        collective = vasp_jumps.collective()
        matrix = collective.matrix()
        assert matrix.shape == (1, 1)
        assert matrix[0, 0] == 1587

    def test_multiple_collective(self, vasp_jumps):
        collective = vasp_jumps.collective()
        jumps, counts = collective.multiple_collective()
        assert jumps.shape == (1261, 2)
        assert counts.sum() == 1587
        assert np.all(jumps[::250][:3] == np.array([[(0, 94), (
            2, 58)], [(9, 17), (64, 42)], [(20, 56), (64, 42)]],
                                                   dtype=[('start',
                                                           int), ('stop',
                                                                  int)]))
