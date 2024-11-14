from __future__ import annotations

import pytest


@pytest.vaspxml_available  # type: ignore
def test_rdf(vasp_rdf_data):
    expected_states = {'~>48h', '@48h', '48h->48h'}
    expected_symbols = {'Li', 'P', 'S', 'Br'}

    assert isinstance(vasp_rdf_data, dict)

    for state, rdfs in vasp_rdf_data.items():
        assert state in expected_states
        for rdf in rdfs:
            assert rdf.label in expected_symbols
            assert len(rdf.x) == 51
            assert len(rdf.y) == 51
