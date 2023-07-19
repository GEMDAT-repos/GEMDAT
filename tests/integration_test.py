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
    equilibration_steps = 74000
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
    assert vol.sum() == 3540000


@vaspxml_available
def test_volume_cartesian(gemdat_results):
    data, extras = gemdat_results

    vol = trajectory_to_volume(lattice=data.lattice,
                               coords=extras.diff_coords,
                               resolution=0.2,
                               cartesian=True)

    assert isinstance(vol, np.ndarray)
    assert vol.shape == (101, 51, 52)
    assert vol.sum() == 3540000


@vaspxml_available
def test_tracer(gemdat_results):
    _, extras = gemdat_results

    assert isclose(extras.particle_density, 2.4557e28, rel_tol=1e-4)
    assert isclose(extras.mol_per_liter, 40.777, rel_tol=1e-4)
    assert isclose(extras.tracer_diff, 1.3524e-09, rel_tol=1e-4)
    assert isclose(extras.tracer_conduc, 94.995, rel_tol=1e-4)


@vaspxml_available
def test_sites(gemdat_results, structure):
    data, extras = gemdat_results

    sites = SitesData(structure)
    sites.calculate_all(data=data, extras=extras)

    assert extras.diff_coords.shape == (73750, 48, 3)

    assert sites.atom_sites.shape == (73750, 48)
    assert sites.atom_sites.sum() == 9085325
    assert sites.atom_sites_to.shape == (73750, 48)
    assert sites.atom_sites_to.sum() == 84790940
    assert sites.atom_sites_from.shape == (73750, 48)
    assert sites.atom_sites_from.sum() == 85358127

    assert sites.all_transitions.shape == (1337, 5)

    assert sites.transitions.shape == (48, 48)

    assert sites.transitions_parts.shape == (extras.n_parts, 48, 48)
    assert np.sum(sites.transitions_parts[0]) == 134
    assert np.sum(sites.transitions_parts[9]) == 142

    assert sites.occupancy[0] == 1748
    assert sites.occupancy[43] == 6350

    assert len(sites.occupancy_parts) == extras.n_parts

    assert sites.occupancy_parts[0][0] == 244
    assert sites.occupancy_parts[0][43] == 1231
    assert sites.occupancy_parts[9][0] == 89
    assert sites.occupancy_parts[9][43] == 391

    assert sites.sites_occupancy == {'Li48h': 0.1477180790960452}

    assert len(sites.occupancy_parts) == extras.n_parts
    assert sites.sites_occupancy_parts[0] == {'Li48h': 0.15146045197740113}
    assert sites.sites_occupancy_parts[9] == {'Li48h': 0.14838418079096044}

    # These appear to be the same in the matlab code
    # https://github.com/GEMDAT-repos/GEMDAT/issues/35
    assert sites.atom_locations == sites.sites_occupancy
    assert sites.atom_locations_parts == sites.sites_occupancy_parts

    assert sites.n_jumps == 1337

    assert isinstance(sites.rates, dict)
    assert len(sites.rates) == 1

    rates, rates_std = sites.rates[('Li48h', 'Li48h')]
    assert isclose(rates, 188841807909.6045)
    assert isclose(rates_std, 16686205789.490553)

    assert isinstance(sites.activation_energies, dict)
    assert len(sites.activation_energies) == 1

    e_act, e_act_std = sites.activation_energies[('Li48h', 'Li48h')]
    assert isclose(e_act, 0.1485147872457603, rel_tol=1e-6)
    assert isclose(e_act_std, 0.005017914800280739, rel_tol=1e-6)

    assert isclose(sites.jump_diffusivity, 3.113091008202005e-09, rel_tol=1e-6)
    assert isclose(sites.correlation_factor, 0.4344246989844385, rel_tol=1e-6)

    assert sites.n_solo_jumps == 987
    assert sites.coll_count == 437
    assert isclose(sites.solo_frac, 0.7382198952879581, rel_tol=1e-4)

    assert len(sites.collective) == 437
    assert sites.collective[0] == (130, 331)
    assert sites.collective[-1] == (1305, 498)

    assert len(sites.coll_jumps) == 437
    assert sites.coll_jumps[0] == ((14, 42), (29, 1))
    assert sites.coll_jumps[-1] == ((26, 38), (46, 10))

    assert sites.coll_matrix.shape == (1, 1)
    assert sites.coll_matrix[0, 0] == 437

    assert sites.multi_coll.sum() == 174380


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

    expected_states = {'~>Li48h', '@Li48h', 'Li48h->Li48h'}
    expected_symbols = set(data.structure.symbol_set)

    assert isinstance(rdfs, dict)

    for state, rdf in rdfs.items():
        assert state in expected_states
        assert set(rdf.keys()) == expected_symbols
        assert all(len(arr) == 51 for arr in rdf.values())
