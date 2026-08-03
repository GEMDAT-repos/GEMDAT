from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pymatgen.core import Lattice, Species, Structure

from gemdat.transitions import (
    SiteRadius,
    _calculate_atom_states,
    _calculate_transitions_matrix,
)


def test_transitions_matrix():
    a = np.array(
        [
            [0, 0, 1, 10],
            [0, 1, 0, 20],
            [1, 2, 1, 5],
            [1, 1, 0, 6],
            [2, 2, 0, 30],
        ]
    )
    a = pd.DataFrame(data=a, columns=['atom index', 'start site', 'destination site', 'time'])

    n = 3
    transitions = _calculate_transitions_matrix(a, n_sites=n)

    assert transitions.shape == (n, n)
    assert transitions.dtype == int
    np.testing.assert_equal(transitions, np.array([[0, 1, 0], [2, 0, 0], [1, 1, 0]]))


@pytest.fixture
def sites():
    return Structure(
        lattice=Lattice(np.eye(3)),
        species=['Li', 'Li'],
        coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        labels=['A', 'B'],
    )


def test_site_radius_from_given_radius_rejects_variable_lattice(
    sites, variable_lattice_trajectory
):
    with pytest.raises(NotImplementedError, match='variable lattice'):
        SiteRadius.from_given_radius(
            trajectory=variable_lattice_trajectory,
            sites=sites,
            radius=0.1,
            inner_fraction=1.0,
        )


def test_site_radius_from_vibration_amplitude_rejects_variable_lattice(
    sites, variable_lattice_trajectory
):
    with pytest.raises(NotImplementedError, match='variable lattice'):
        SiteRadius.from_vibration_amplitude(
            trajectory=variable_lattice_trajectory,
            sites=sites,
            vibration_amplitude=0.1,
        )


def test_calculate_atom_states_rejects_variable_lattice(sites, variable_lattice_trajectory):
    with pytest.raises(NotImplementedError, match='variable lattice'):
        _calculate_atom_states(
            sites=sites,
            trajectory=variable_lattice_trajectory,
            site_radius={'A': 0.1, 'B': 0.1},
            site_inner_fraction={'A': 1.0, 'B': 1.0},
        )


@pytest.fixture
def jumps():
    """Jumps built from a constant-lattice trajectory.

    Transitions cannot be built from a variable lattice at all, so the
    guards downstream of it are exercised by flipping the flag
    afterwards.
    """
    from gemdat.trajectory import Trajectory

    n_steps = 40
    coords = np.zeros((n_steps, 2, 3))
    coords[:, 0, 0] = np.linspace(0.05, 0.95, n_steps)
    coords[:, 1] = [0.5, 0.5, 0.5]

    trajectory = Trajectory(
        species=[Species('Li'), Species('S')],
        coords=coords,
        lattice=np.eye(3) * 5,
        constant_lattice=True,
        metadata={'temperature': 300},
        time_step=1e-12,
    )
    sites = Structure(
        lattice=Lattice(np.eye(3) * 5),
        species=['Li', 'Li'],
        coords=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        labels=['A', 'B'],
    )
    transitions = trajectory.transitions_between_sites(
        sites=sites, floating_specie='Li', site_radius=1.0
    )
    return transitions.jumps()


def test_jump_diffusivity_rejects_variable_lattice(jumps):
    assert jumps.jump_diffusivity(dimensions=3) > 0

    jumps.trajectory.constant_lattice = False
    with pytest.raises(NotImplementedError, match='variable lattice'):
        jumps.jump_diffusivity(dimensions=3)


def test_collective_rejects_variable_lattice(jumps):
    jumps.trajectory.constant_lattice = False

    with pytest.raises(NotImplementedError, match='variable lattice'):
        jumps.collective()


def test_radial_distribution_rejects_variable_lattice(jumps):
    from gemdat.rdf import radial_distribution

    jumps.transitions.trajectory.constant_lattice = False

    with pytest.raises(NotImplementedError, match='variable lattice'):
        radial_distribution(transitions=jumps.transitions, floating_specie='Li')
