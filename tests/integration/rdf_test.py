import pytest


@pytest.vaspxml_available
def test_rdf(vasp_rdf_data):
    expected_states = {'~>48h', '@48h', '48h->48h'}
    expected_symbols = {'Li', 'P', 'S', 'Br'}

    assert isinstance(vasp_rdf_data, dict)

    for state, rdfs in vasp_rdf_data.items():
        assert state in expected_states
        assert set(rdfs.keys()) == expected_symbols
        assert all(len(rdf.y) == 51 for rdf in rdfs.values())
        assert all(len(rdf.x) == 51 for rdf in rdfs.values())
