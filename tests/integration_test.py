"""
Run integration test with:

VASP_XML=/home/stef/md-analysis-matlab-example/vasprun.xml pytest
"""

from math import isclose
from pathlib import Path

import numpy as np
import pytest
from gemdat import SitesData
from gemdat.calculate import calculate_all
from gemdat.io import load_known_material
from gemdat.rdf import calculate_rdfs
from gemdat.trajectory import Trajectory
from gemdat.volume import trajectory_to_volume

DATA_DIR = Path(__file__).parent / 'data'
VASP_XML = DATA_DIR / 'short_simulation' / 'vasprun.xml'

vaspxml_available = pytest.mark.skipif(
    not VASP_XML.exists(),
    reason=
    ('Simulation data from vasprun.xml example is required for this test. '
     'Run `git submodule init`/`update`, and extract using `tar -C tests/data/short_simulation '
     '-xjf tests/data/short_simulation/vasprun.xml.bz2`'))


@pytest.fixture
def gemdat_results():
    trajectory = Trajectory.from_vasprun(VASP_XML)
    trajectory = trajectory[1250:]

    extras = calculate_all(
        trajectory,
        diffusing_element='Li',
        z_ion=1,
        diffusion_dimensions=3,
    )

    return (trajectory, extras)


@pytest.fixture
def gemdat_results_subset():
    trajectory = Trajectory.from_vasprun(VASP_XML)
    # Reduced number of time steps for slow calculations
    trajectory = trajectory[4000:]

    extras = calculate_all(
        trajectory,
        diffusing_element='Li',
        z_ion=1,
        diffusion_dimensions=3,
        n_parts=1,
    )

    return (trajectory, extras)


@pytest.fixture
def structure():
    return load_known_material('argyrodite', supercell=(2, 1, 1))


@vaspxml_available
def test_volume(gemdat_results):
    trajectory, extras = gemdat_results

    diff_trajectory = trajectory.filter(extras.diffusing_element)

    vol = trajectory_to_volume(trajectory=diff_trajectory, resolution=0.2)

    assert isinstance(vol, np.ndarray)
    assert vol.shape == (101, 51, 51)
    assert vol.sum() == len(diff_trajectory.species) * len(diff_trajectory)


@vaspxml_available
def test_volume_cartesian(gemdat_results):
    trajectory, extras = gemdat_results

    diff_trajectory = trajectory.filter(extras.diffusing_element)

    vol = trajectory_to_volume(trajectory=diff_trajectory,
                               resolution=0.2,
                               cartesian=True)

    assert isinstance(vol, np.ndarray)
    # assert vol.shape == (101, 51, 52)
    assert vol.sum() == len(diff_trajectory.species) * len(diff_trajectory)


@vaspxml_available
def test_tracer(gemdat_results):
    _, extras = gemdat_results

    assert isclose(extras.particle_density, 2.4557e28, rel_tol=1e-4)
    assert isclose(extras.mol_per_liter, 40.777, rel_tol=1e-4)
    assert isclose(extras.tracer_diff, 1.5706e-09, rel_tol=1e-4)
    assert isclose(extras.tracer_conduc, 110.322, rel_tol=1e-4)


@vaspxml_available
def test_sites(gemdat_results, structure):
    trajectory, extras = gemdat_results

    sites = SitesData(structure)
    sites.calculate_all(trajectory=trajectory, extras=extras)

    n_steps = extras.n_steps
    n_diffusing = sum(
        [sp.symbol == extras.diffusing_element for sp in trajectory.species])
    n_sites = sites.n_sites

    assert sites.atom_sites.shape == (n_steps, n_diffusing)
    assert sites.atom_sites.sum() == 6157517
    assert sites.atom_sites_to.shape == (n_steps, n_diffusing)
    assert sites.atom_sites_to.sum() == 8170281
    assert sites.atom_sites_from.shape == (n_steps, n_diffusing)
    assert sites.atom_sites_from.sum() == 8150432

    assert sites.all_transitions.shape == (456, 5)

    assert sites.transitions.shape == (n_sites, n_sites)

    assert sites.transitions_parts.shape == (extras.n_parts, n_sites, n_sites)
    assert np.sum(sites.transitions_parts[0]) == 37
    assert np.sum(sites.transitions_parts[9]) == 38

    assert sites.occupancy[0] == 1717
    assert sites.occupancy[43] == 542

    assert len(sites.occupancy_parts) == extras.n_parts

    assert sites.occupancy_parts[0][0] == 56
    assert sites.occupancy_parts[0][42] == 36
    assert sites.occupancy_parts[9][0] == 62
    assert sites.occupancy_parts[9][42] == 177

    assert isclose(sites.sites_occupancy['48h'], 0.380628, rel_tol=1e-4)

    assert len(sites.sites_occupancy_parts) == extras.n_parts
    assert isclose(sites.sites_occupancy_parts[0]['48h'],
                   0.37694,
                   rel_tol=1e-4)
    assert isclose(sites.sites_occupancy_parts[9]['48h'],
                   0.37027,
                   rel_tol=1e-4)

    assert isclose(sites.atom_locations['48h'], 0.761255, rel_tol=1e-4)

    assert len(sites.atom_locations_parts) == extras.n_parts
    assert isclose(sites.atom_locations_parts[0]['48h'], 0.75389, rel_tol=1e-4)
    assert isclose(sites.atom_locations_parts[9]['48h'], 0.74055, rel_tol=1e-4)

    assert sites.n_jumps == 456

    assert isinstance(sites.rates, dict)
    assert len(sites.rates) == 1

    rates, rates_std = sites.rates[('48h', '48h')]
    assert isclose(rates, 1266666666666.6665)
    assert isclose(rates_std, 145932505961.81885)

    assert isinstance(sites.activation_energies, dict)
    assert len(sites.activation_energies) == 1

    e_act, e_act_std = sites.activation_energies[('48h', '48h')]
    assert isclose(e_act, 0.1299544, rel_tol=1e-6)
    assert isclose(e_act_std, 0.006650821, rel_tol=1e-6)

    assert isclose(sites.jump_diffusivity, 9.436645866723717e-9, rel_tol=1e-6)
    assert isclose(sites.correlation_factor, 0.1664378436418788, rel_tol=1e-6)

    assert sites.n_solo_jumps == 1974
    assert sites.coll_count == 1313
    assert isclose(sites.solo_frac, 4.328947, rel_tol=1e-4)

    assert len(sites.collective) == 1313

    assert sites.collective[0] == (158, 390)
    assert sites.collective[-1] == (350, 389)

    assert len(sites.coll_jumps) == 1313
    assert sites.coll_jumps[0] == ((74, 8), (41, 67))
    assert sites.coll_jumps[-1] == ((15, 77), (21, 45))

    assert sites.coll_matrix.shape == (1, 1)
    assert sites.coll_matrix[0, 0] == 1313

    assert sites.multi_coll.sum() == 452717


@vaspxml_available
def test_rdf(gemdat_results_subset, structure):
    trajectory, extras = gemdat_results_subset

    structure = load_known_material('argyrodite')

    sites = SitesData(structure)
    sites.calculate_all(trajectory=trajectory, extras=extras)

    rdfs = calculate_rdfs(
        trajectory=trajectory,
        sites=sites,
        species=extras.diffusing_element,
        max_dist=5,
    )

    expected_states = {'~>48h', '@48h', '48h->48h'}
    expected_symbols = set(trajectory.get_structure(0).symbol_set)

    assert isinstance(rdfs, dict)

    for state, rdf in rdfs.items():
        assert state in expected_states
        assert set(rdf.keys()) == expected_symbols
        assert all(len(arr) == 51 for arr in rdf.values())
