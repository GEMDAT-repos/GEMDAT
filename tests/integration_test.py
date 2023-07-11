"""
Run integration test with:

VASP_XML=/run/media/stef/Scratch/md-analysis-matlab-example/vasprun.xml pytest
"""

import os

import numpy as np
import pytest
from gemdat import SimulationData

VASP_XML = os.environ.get('VASP_XML')

vaspxml_available = pytest.mark.skipif(
    VASP_XML is None,
    reason='vasprun.xml test sim_data is required for this test.')


@pytest.fixture
def gemdat_results():
    equilibration_steps = 1250
    diffusing_element = 'Li'
    diffusion_dimensions = 3
    z_ion = 1

    data = SimulationData.from_vasprun(VASP_XML, cache='vasprun.xml.cache')

    extras = data.calculate_all(
        equilibration_steps=equilibration_steps,
        diffusing_element=diffusing_element,
        z_ion=z_ion,
        diffusion_dimensions=diffusion_dimensions,
    )

    return (data, extras)


@vaspxml_available
def test_tracer(gemdat_results):
    from math import isclose

    _, extras = gemdat_results

    assert isclose(extras.particle_density, 2.4557e28, rel_tol=1e-4)
    assert isclose(extras.mol_per_liter, 40.777, rel_tol=1e-4)
    assert isclose(extras.tracer_diff, 1.3524e-09, rel_tol=1e-4)
    assert isclose(extras.tracer_conduc, 94.995, rel_tol=1e-4)


@vaspxml_available
def test_sites(gemdat_results):
    from gemdat.io import load_known_material
    from gemdat.sites import SitesData

    data, extras = gemdat_results

    structure = load_known_material('argyrodite')

    sites = SitesData(structure)
    sites.calculate_all(data=data, extras=extras)

    assert extras.diff_coords.shape == (73750, 48, 3)

    assert sites.atom_sites.shape == (73750, 48)
    assert sites.atom_sites.sum() == 9228360
    assert sites.atom_sites_to.shape == (73750, 48)
    assert sites.atom_sites_to.sum() == 84831031
    assert sites.atom_sites_from.shape == (73750, 48)
    assert sites.atom_sites_from.sum() == 85351052

    assert sites.all_transitions.shape == (1336, 5)

    assert sites.transitions.shape == (48, 48)

    assert sites.transitions_parts.shape == (extras.n_parts, 48, 48)
    assert np.sum(sites.transitions_parts[0]) == 134
    assert np.sum(sites.transitions_parts[9]) == 142

    assert sites.occupancy[0] == 1706
    assert sites.occupancy[43] == 6350

    assert len(sites.occupancy_parts) == extras.n_parts

    assert sites.occupancy_parts[0][0] == 241
    assert sites.occupancy_parts[0][43] == 1231
    assert sites.occupancy_parts[9][0] == 87
    assert sites.occupancy_parts[9][43] == 391

    assert sites.sites_occupancy == {'Li48h': 0.14825282485875707}

    assert len(sites.occupancy_parts) == extras.n_parts
    assert sites.sites_occupancy_parts[0] == {'Li48h': 0.15155367231638417}
    assert sites.sites_occupancy_parts[9] == {'Li48h': 0.14890395480225987}

    # These appear to be the same in the matlab code
    # https://github.com/GEMDAT-repos/GEMDAT/issues/35
    assert sites.atom_locations == sites.sites_occupancy
    assert sites.atom_locations_parts == sites.sites_occupancy_parts

    assert sites.n_jumps == 1336

    assert isinstance(sites.rates, dict)
    assert len(sites.rates) == 1
    assert sites.rates['Li48h',
                       'Li48h'] == (188700564971.7514, 16674910449.098799)

    assert isinstance(sites.activation_energies, dict)
    assert len(sites.activation_energies) == 1
    assert sites.activation_energies[('Li48h',
                                      'Li48h')] == (0.14859184952052334,
                                                    0.005030984063924976)
