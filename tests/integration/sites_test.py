"""
Run integration test with:

VASP_XML=/home/stef/md-analysis-matlab-example/vasprun.xml pytest
"""

from math import isclose

import numpy as np
import pytest


@pytest.vaspxml_available
class TestSites:
    n_parts = 10
    diffusing_element = 'Li'

    def test_atom_sites(self, vasp_sites, vasp_traj):
        n_steps = len(vasp_traj)
        n_diffusing = vasp_sites.n_floating

        transitions = vasp_sites.transitions

        assert isclose(transitions._dist_close, 0.928496)

        assert transitions.states.shape == (n_steps, n_diffusing)
        assert transitions.states.sum() == 6154859
        assert transitions.states_next().shape == (n_steps, n_diffusing)
        assert transitions.states_next().sum() == 8172006
        assert transitions.states_prev().shape == (n_steps, n_diffusing)
        assert transitions.states_prev().sum() == 8148552

    def test_all_transitions(self, vasp_sites):
        transitions = vasp_sites.transitions
        assert transitions.events.shape == (450, 5)
        assert transitions.matrix().shape == (vasp_sites.n_sites,
                                              vasp_sites.n_sites)

    def test_transitions_parts(self, vasp_sites):
        n_sites = vasp_sites.n_sites
        assert vasp_sites.transitions_parts.shape == (self.n_parts, n_sites,
                                                      n_sites)
        assert np.sum(vasp_sites.transitions_parts[0]) == 37
        assert np.sum(vasp_sites.transitions_parts[9]) == 38

    def test_occupancy(self, vasp_sites):
        assert vasp_sites.transitions.occupancy()[0] == 1704
        assert vasp_sites.transitions.occupancy()[43] == 542

    def test_occupancy_parts(self, vasp_sites):
        assert len(vasp_sites.occupancy_parts) == self.n_parts

        assert vasp_sites.occupancy_parts[0][0] == 56
        assert vasp_sites.occupancy_parts[0][42] == 36
        assert vasp_sites.occupancy_parts[9][0] == 62
        assert vasp_sites.occupancy_parts[9][42] == 177

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
        assert isclose(e_act, 0.130754, rel_tol=1e-6)
        assert isclose(e_act_std, 0.0063201, rel_tol=1e-6)

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
        assert len(collective.collective) == 1280

        assert collective.collective[0] == (158, 384)
        assert collective.collective[-1] == (348, 383)

    def test_coll_jumps(self, vasp_sites):
        collective = vasp_sites.collective()

        assert len(collective.coll_jumps) == 1280
        assert collective.coll_jumps[0] == ((74, 8), (41, 67))
        assert collective.coll_jumps[-1] == ((15, 77), (21, 45))

    def test_collective_matrix(self, vasp_sites):
        collective = vasp_sites.collective()
        assert collective.matrix().shape == (1, 1)
        assert collective.matrix()[0, 0] == 1280

    def test_multiple_collective(self, vasp_sites):
        collective = vasp_sites.collective()
        assert collective.multiple_collective().sum() == 434227
