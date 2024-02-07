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

from gemdat.transitions import _compute_site_radius


@pytest.vaspxml_available
class TestTransitions:  # type: ignore
    n_parts = 10
    diffusing_element = 'Li'

    def test_site_radius(self, vasp_traj, structure):
        site_radius = _compute_site_radius(
            trajectory=vasp_traj,
            sites=structure,
            vibration_amplitude=0.5204,
        )
        assert isclose(site_radius, 0.9284961123176741)

    def test_atom_sites(self, vasp_traj, vasp_transitions):
        n_steps = len(vasp_traj)
        n_diffusing = 48

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
