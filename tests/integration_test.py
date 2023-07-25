"""
Run integration test with:

VASP_XML=/home/stef/md-analysis-matlab-example/vasprun.xml pytest
"""

import os
from math import isclose

import numpy as np
import pytest
from gemdat import SimulationData, SitesData
from gemdat.io import load_known_material
from gemdat.rdf import calculate_rdfs
from gemdat.volume import trajectory_to_volume

VASP_XML = os.environ.get('VASP_XML')

vaspxml_available = pytest.mark.skipif(
    VASP_XML is None,
    reason='Simulation data from vasprun.xml example is required for this test.'
)


@pytest.fixture
def gemdat_results():
    equilibration_steps = 1250
    diffusing_element = 'Li'
    diffusion_dimensions = 3
    z_ion = 1

    data = SimulationData.from_vasprun(VASP_XML)

    extras = data.calculate_all(
        equilibration_steps=equilibration_steps,
        diffusing_element=diffusing_element,
        z_ion=z_ion,
        diffusion_dimensions=diffusion_dimensions,
    )

    return (data, extras)


@pytest.fixture
def gemdat_results_subset():
    # Reduced number of time steps for slow calculations
    equilibration_steps = 1250
    diffusing_element = 'Li'
    diffusion_dimensions = 3
    z_ion = 1

    data = SimulationData.from_vasprun(VASP_XML)

    extras = data.calculate_all(
        equilibration_steps=equilibration_steps,
        diffusing_element=diffusing_element,
        z_ion=z_ion,
        diffusion_dimensions=diffusion_dimensions,
    )

    return (data, extras)


@pytest.fixture
def structure():
    return load_known_material('argyrodite')


@vaspxml_available
def test_volume(gemdat_results):
    data, extras = gemdat_results

    vol = trajectory_to_volume(lattice=data.lattice,
                               coords=extras.diff_coords,
                               resolution=0.2)

    assert isinstance(vol, np.ndarray)
    assert vol.shape == (101, 51, 51)
    assert vol.sum() == 180000


@vaspxml_available
def test_volume_cartesian(gemdat_results):
    data, extras = gemdat_results

    vol = trajectory_to_volume(lattice=data.lattice,
                               coords=extras.diff_coords,
                               resolution=0.2,
                               cartesian=True)

    assert isinstance(vol, np.ndarray)
    assert vol.shape == (101, 51, 52)
    assert vol.sum() == 180000


@vaspxml_available
def test_tracer(gemdat_results):
    _, extras = gemdat_results

    assert isclose(extras.particle_density, 2.4557e28, rel_tol=1e-4)
    assert isclose(extras.mol_per_liter, 40.777, rel_tol=1e-4)
    assert isclose(extras.tracer_diff, 1.5706e-09, rel_tol=1e-4)
    assert isclose(extras.tracer_conduc, 110.322, rel_tol=1e-4)


@vaspxml_available
def test_sites(gemdat_results, structure):
    data, extras = gemdat_results

    sites = SitesData(structure)
    sites.calculate_all(data=data, extras=extras)

    assert extras.diff_coords.shape == (3750, 48, 3)

    assert sites.atom_sites.shape == (3750, 48)
    assert sites.atom_sites.sum() == 500416
    assert sites.atom_sites_to.shape == (3750, 48)
    assert sites.atom_sites_to.sum() == 3091708
    assert sites.atom_sites_from.shape == (3750, 48)
    assert sites.atom_sites_from.sum() == 2994256

    assert sites.all_transitions.shape == (40, 5)

    assert sites.transitions.shape == (48, 48)

    assert sites.transitions_parts.shape == (extras.n_parts, 48, 48)
    assert np.sum(sites.transitions_parts[0]) == 2
    assert np.sum(sites.transitions_parts[9]) == 3

    assert sites.occupancy[0] == 207
    assert sites.occupancy[43] == 216

    assert len(sites.occupancy_parts) == extras.n_parts

    assert sites.occupancy_parts[0][0] == 42
    assert sites.occupancy_parts[0][42] == 204
    assert sites.occupancy_parts[9][0] == 7
    assert sites.occupancy_parts[9][42] == 33

    assert isclose(sites.sites_occupancy['48h'], 0.160072, rel_tol=1e-4)

    assert len(sites.occupancy_parts) == extras.n_parts
    assert sites.sites_occupancy_parts[0] == {'48h': 0.1525}
    assert isclose(sites.sites_occupancy_parts[9]['48h'],
                   0.137666,
                   rel_tol=1e-4)

    # These appear to be the same in the matlab code
    # https://github.com/GEMDAT-repos/GEMDAT/issues/35
    assert sites.atom_locations == sites.sites_occupancy
    assert sites.atom_locations_parts == sites.sites_occupancy_parts

    assert sites.n_jumps == 40

    assert isinstance(sites.rates, dict)
    assert len(sites.rates) == 1

    rates, rates_std = sites.rates[('48h', '48h')]
    assert isclose(rates, 111111111111.1111)
    assert isclose(rates_std, 58560697410.525536)

    assert isinstance(sites.activation_energies, dict)
    assert len(sites.activation_energies) == 1

    e_act, e_act_std = sites.activation_energies[('48h', '48h')]
    assert isclose(e_act, 0.18702541420508717, rel_tol=1e-6)
    assert isclose(e_act_std, 0.04058155516885685, rel_tol=1e-6)

    assert isclose(sites.jump_diffusivity,
                   1.6561040438301754e-09,
                   rel_tol=1e-6)
    assert isclose(sites.correlation_factor, 0.9483794180207937, rel_tol=1e-6)

    assert sites.n_solo_jumps == 17
    assert sites.coll_count == 5
    assert isclose(sites.solo_frac, 0.425, rel_tol=1e-4)

    assert len(sites.collective) == 5
    assert sites.collective[0] == (25, 1)
    assert sites.collective[-1] == (11, 20)

    assert len(sites.coll_jumps) == 5
    assert sites.coll_jumps[0] == ((38, 7), (42, 10))
    assert sites.coll_jumps[-1] == ((23, 34), (19, 3))

    assert sites.coll_matrix.shape == (1, 1)
    assert sites.coll_matrix[0, 0] == 5

    assert sites.multi_coll.sum() == 30


@vaspxml_available
def test_rdf(gemdat_results_subset, structure):
    data, extras = gemdat_results_subset

    structure = load_known_material('argyrodite')

    sites = SitesData(structure)
    sites.calculate_all(data=data, extras=extras)

    rdfs = calculate_rdfs(
        data=data,
        sites=sites,
        diff_coords=extras.diff_coords,
        n_steps=extras.n_steps,
        equilibration_steps=extras.equilibration_steps,
        max_dist=5,
    )

    expected_states = {'~>48h', '@48h', '48h->48h'}
    expected_symbols = set(data.structure.symbol_set)

    assert isinstance(rdfs, dict)

    for state, rdf in rdfs.items():
        assert state in expected_states
        assert set(rdf.keys()) == expected_symbols
        assert all(len(arr) == 51 for arr in rdf.values())
