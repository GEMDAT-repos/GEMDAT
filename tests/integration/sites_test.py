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
        assert transitions.events.shape == (450, 5)
        assert_allclose(
            events[::100],
            np.array([
                [0, 94, 0, 228, 284],
                [9, 60, 68, 2692, 2946],
                [18, 24, 54, 3435, 3646],
                [31, 59, 51, 1240, 1536],
                [41, 41, 67, 3633, 3667],
            ]))

        matrix = transitions.matrix()
        assert matrix.shape == (vasp_sites.n_sites, vasp_sites.n_sites)
        assert matrix.sum() == 450
        assert_allclose(
            np.argwhere(matrix)[::50],
            np.array([[0, 94], [26, 10], [48, 32], [74, 8]]))

    def test_transitions_parts(self, vasp_sites):
        n_sites = vasp_sites.n_sites
        tp = vasp_sites.transitions_parts

        assert tp.shape == (self.n_parts, n_sites, n_sites)
        assert tp.sum() == 450
        assert_allclose(
            np.argwhere(tp)[::100],
            np.array([
                [0, 2, 66],
                [2, 43, 51],
                [5, 4, 90],
                [7, 23, 31],
            ]))

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
        assert len(vasp_sites.atom_locations_parts) == self.n_parts
        assert isclose(vasp_sites.atom_locations_parts[0]['48h'],
                       0.755111,
                       rel_tol=1e-4)
        assert isclose(vasp_sites.atom_locations_parts[9]['48h'],
                       0.738444,
                       rel_tol=1e-4)

    def test_n_jumps(self, vasp_sites):
        assert vasp_sites.n_solo_jumps == 1922
        assert vasp_sites.n_jumps == 450
        assert isclose(vasp_sites.solo_fraction, 4.2711, rel_tol=1e-4)

    def test_rates(self, vasp_sites):
        assert isinstance(vasp_sites.rates(), dict)
        assert len(vasp_sites.rates()) == 1

        rates, rates_std = vasp_sites.rates()[('48h', '48h')]
        assert isclose(rates, 1249999999999.9998)
        assert isclose(rates_std, 137337009020.29002)

    def test_activation_energies(self, vasp_sites):
        assert isinstance(vasp_sites.activation_energies(), dict)
        assert len(vasp_sites.activation_energies()) == 1

        e_act, e_act_std = vasp_sites.activation_energies()[('48h', '48h')]
        assert isclose(e_act, 0.1311852, rel_tol=1e-6)
        assert isclose(e_act_std, 0.00596132, rel_tol=1e-6)

    def test_jump_diffusivity(self, vasp_sites):
        assert isclose(vasp_sites.jump_diffusivity(3),
                       9.220713700212185e-09,
                       rel_tol=1e-6)

    def test_correlation_factor(self, vasp_sites):
        tracer_diff = vasp_sites.metrics.tracer_diffusivity(dimensions=3)
        correlation_factor = tracer_diff / vasp_sites.jump_diffusivity(
            dimensions=3)
        assert isclose(correlation_factor, 0.1703355120150192, rel_tol=1e-6)

    def test_collective(self, vasp_sites):
        collective = vasp_sites.collective()
        cc = collective.collective

        assert len(cc) == 1280
        assert cc[::1000] == [(158, 384), (33, 113)]

    def test_coll_jumps(self, vasp_sites):
        collective = vasp_sites.collective()
        coll_jumps = collective.coll_jumps

        assert len(coll_jumps) == 1280
        assert coll_jumps[::1000] == [((74, 8), (41, 67)), ((6, 88), (62, 18))]

    def test_collective_matrix(self, vasp_sites):
        collective = vasp_sites.collective()
        matrix = collective.matrix()
        assert matrix.shape == (1, 1)
        assert matrix[0, 0] == 1280

    def test_multiple_collective(self, vasp_sites):
        collective = vasp_sites.collective()
        mc = collective.multiple_collective()
        assert mc.shape == (2112, )
        assert mc.sum() == 434227
        assert_allclose(mc[::250],
                        np.array([0, 50, 98, 137, 187, 232, 284, 347, 420]))
