"""
Run integration test with:

VASP_XML=/home/stef/md-analysis-matlab-example/vasprun.xml pytest
"""

from math import isclose
from pathlib import Path

import numpy as np
import pytest
from gemdat import SitesData, Trajectory, Vibration
from gemdat.io import load_known_material
from gemdat.rdf import calculate_rdfs
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
    equilibration_steps = 1250

    trajectory = Trajectory.from_vasprun(VASP_XML)
    trajectory = trajectory[equilibration_steps:]
    trajectory_li = trajectory.filter('Li')

    return trajectory_li


@pytest.fixture
def gemdat_results_subset():
    # Reduced number of time steps for slow calculations
    equilibration_steps = 4000

    trajectory = Trajectory.from_vasprun(VASP_XML)
    trajectory = trajectory[equilibration_steps:]
    trajectory_li = trajectory.filter('Li')

    return trajectory_li


@pytest.fixture
def structure():
    return load_known_material('argyrodite', supercell=(2, 1, 1))


@vaspxml_available
def test_volume(gemdat_results):
    trajectory = gemdat_results

    vol = trajectory_to_volume(lattice=trajectory.get_lattice(),
                               coords=trajectory.coords,
                               resolution=0.2)

    assert isinstance(vol, np.ndarray)
    assert vol.shape == (101, 51, 51)
    assert vol.sum() == len(trajectory.species) * len(trajectory)


@vaspxml_available
def test_volume_cartesian(gemdat_results):
    trajectory = gemdat_results

    vol = trajectory_to_volume(lattice=trajectory.get_lattice(),
                               coords=trajectory.coords,
                               resolution=0.2,
                               cartesian=True)

    assert isinstance(vol, np.ndarray)
    assert vol.shape == (101, 51, 52)
    assert vol.sum() == len(trajectory.species) * len(trajectory)


@vaspxml_available
def test_tracer(gemdat_results):
    trajectory = gemdat_results

    assert isclose(trajectory.particle_density(), 2.4557e28, rel_tol=1e-4)
    assert isclose(trajectory.mol_per_liter, 40.777, rel_tol=1e-4)
    assert isclose(trajectory.tracer_diffusivity(), 1.5706e-09, rel_tol=1e-4)
    assert isclose(trajectory.tracer_conductivity(), 110.322, rel_tol=1e-4)


@vaspxml_available
def test_sites(gemdat_results, structure):
    trajectory = gemdat_results

    sites = SitesData(structure, trajectory, Vibration(trajectory, fs=1.0))

    n_steps = len(trajectory)
    n_diffusing = len(trajectory.species)
    n_sites = sites.n_sites

    assert trajectory.coords.shape == (n_steps, n_diffusing, 3)

    assert sites.atom_sites().shape == (n_steps, n_diffusing)
    assert sites.atom_sites().sum() == 6154859
    assert sites.atom_sites_to().shape == (n_steps, n_diffusing)
    assert sites.atom_sites_to().sum() == 8172006
    assert sites.atom_sites_from().shape == (n_steps, n_diffusing)
    assert sites.atom_sites_from().sum() == 8148552

    assert sites.all_transitions().shape == (450, 5)

    assert sites.transitions().shape == (n_sites, n_sites)

    assert sites.transitions_parts(10).shape == (10, n_sites, n_sites)
    assert np.sum(sites.transitions_parts(10)[0]) == 33
    assert np.sum(sites.transitions_parts(10)[8]) == 35

    assert sites.occupancy()[0] == 1704
    assert sites.occupancy()[43] == 542

    assert len(sites.occupancy_parts(10)) == 10

    assert sites.occupancy_parts(10)[0][0] == 56
    assert sites.occupancy_parts(10)[0][42] == 36
    assert sites.occupancy_parts(10)[9][0] == 62
    assert sites.occupancy_parts(10)[9][42] == 177

    assert isclose(sites.site_occupancy()['48h'], 0.380628, rel_tol=1e-4)

    assert len(sites.site_occupancy_parts(10)) == 10
    assert isclose(sites.site_occupancy_parts(10)[0]['48h'],
                   0.377555,
                   rel_tol=1e-4)
    assert isclose(sites.site_occupancy_parts(10)[9]['48h'],
                   0.36922,
                   rel_tol=1e-4)

    assert isclose(sites.atom_locations()['48h'], 0.761255, rel_tol=1e-4)

    assert len(sites.atom_locations_parts(10)) == 10
    assert isclose(sites.atom_locations_parts(10)[0]['48h'],
                   0.755111,
                   rel_tol=1e-4)
    assert isclose(sites.atom_locations_parts(10)[9]['48h'],
                   0.738444,
                   rel_tol=1e-4)

    assert sites.n_jumps() == 450

    assert isinstance(sites.rates(10), dict)
    assert len(sites.rates(10)) == 1

    rates, rates_std = sites.rates(10)[('48h', '48h')]
    assert isclose(rates, 936111111111.1111)
    assert isclose(rates_std, 101472356983.68504)

    assert isinstance(sites.activation_energies(10), dict)
    assert len(sites.activation_energies(10)) == 1

    e_act, e_act_std = sites.activation_energies(10)[('48h', '48h')]
    assert isclose(e_act, 0.146967448, rel_tol=1e-4)
    assert isclose(e_act_std, 0.00661589, rel_tol=1e-4)

    assert isclose(sites.jump_diffusivity(),
                   9.220713700212185e-09,
                   rel_tol=1e-6)
    assert isclose(sites.correlation_factor(),
                   0.1703355120150192,
                   rel_tol=1e-6)

    collective, coll_jumps, n_solo_jumps = sites.collective()

    assert n_solo_jumps == 1922
    assert len(coll_jumps) == 1280
    assert isclose(sites.solo_frac(), 4.2711, rel_tol=1e-4)

    assert len(collective) == 1280

    assert collective[0] == (158, 384)
    assert collective[-1] == (348, 383)

    assert len(coll_jumps) == 1280
    assert coll_jumps[0] == ((74, 8), (41, 67))
    assert coll_jumps[-1] == ((15, 77), (21, 45))

    assert sites.collective_matrix().shape == (1, 1)
    assert sites.collective_matrix()[0, 0] == 1280

    assert sites.multiple_collective().sum() == 434227


@vaspxml_available
@pytest.mark.skip('RDF needs to be fixed first')
def test_rdf(gemdat_results_subset, structure):
    trajectory = gemdat_results_subset

    structure = load_known_material('argyrodite')

    sites = SitesData(structure)

    rdfs = calculate_rdfs(
        trajectory=trajectory,
        sites=sites,
        diff_coords=trajectory.coords,
        n_steps=len(trajectory),
        max_dist=5,
    )

    expected_states = {'~>48h', '@48h', '48h->48h'}
    expected_symbols = set(trajectory.get_structure(0).symbol_set)

    assert isinstance(rdfs, dict)

    for state, rdf in rdfs.items():
        assert state in expected_states
        assert set(rdf.keys()) == expected_symbols
        assert all(len(arr) == 51 for arr in rdf.values())
