import pytest
from gemdat.rdf import radial_distribution
from gemdat.sites import SitesData


@pytest.vaspxml_available
def test_rdf(vasp_traj, structure):
    # Shorten trajectory for faster test
    trajectory = vasp_traj[-1000:]

    sites = SitesData(structure=structure,
                      trajectory=trajectory,
                      floating_specie='Li')

    rdfs = radial_distribution(
        trajectory=trajectory,
        sites=sites,
        species='Li',
        max_dist=5,
    )

    expected_states = {'~>48h', '@48h', '48h->48h'}
    expected_symbols = set(trajectory.get_structure(0).symbol_set)

    assert isinstance(rdfs, dict)

    for state, rdf in rdfs.items():
        assert state in expected_states
        assert set(rdf.keys()) == expected_symbols
        assert all(len(arr) == 51 for arr in rdf.values())
