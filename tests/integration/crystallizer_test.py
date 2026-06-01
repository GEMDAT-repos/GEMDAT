from __future__ import annotations

import pytest
from pymatgen.core import Structure

from gemdat.crystallizer import Crystallizer, CrystallizerResult
from gemdat.io import read_cif


@pytest.fixture(scope='module')
def vasp_crystallizer(vasp_traj):
    trajectory = vasp_traj[-250:]
    return Crystallizer.from_trajectory(trajectory, floating_specie='Li')


@pytest.vaspxml_available  # type: ignore
def test_crystallize(vasp_crystallizer):
    result = vasp_crystallizer.crystallize()

    assert isinstance(result, CrystallizerResult)
    assert isinstance(result.structure, Structure)
    assert len(result.structure) > 0
    # P1 (#1) is a valid physical outcome: this fixture is a short, thermally
    # noisy 2x1x1 Li6PS5Br supercell whose S/Br framework is site-disordered in
    # the MD cell, so the time-averaged structure need not be high-symmetry.
    assert result.spacegroup_number >= 1


@pytest.vaspxml_available  # type: ignore
def test_to_cif(vasp_crystallizer, tmp_path):
    filename = tmp_path / 'argyrodite.cif'
    result = vasp_crystallizer.to_cif(filename)

    assert filename.exists()
    assert result.spacegroup_number >= 1

    reread = read_cif(filename)
    assert isinstance(reread, Structure)
    assert len(reread) > 0
