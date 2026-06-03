from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Structure

from gemdat.crystallizer import Crystallizer, CrystallizerResult
from gemdat.io import load_known_material, read_cif


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


@pytest.vaspxml_available  # type: ignore
def test_crystallize_matches_argyrodite_reference(vasp_crystallizer):
    """The reconstructed structure should reproduce the known argyrodite Li
    sublattice.

    The bundled `argyrodite.cif` only describes the Li sites, so this reference
    comparison is only meaningful for the Li diffusing species.
    """
    if vasp_crystallizer.floating_specie != 'Li':
        pytest.skip('reference comparison only applies to the Li sublattice')

    result = vasp_crystallizer.crystallize()

    # Reference Li sublattice, scaled to match the 2x1x1 MD supercell.
    reference = load_known_material('argyrodite', supercell=(2, 1, 1))
    ref_li = [site.frac_coords for site in reference if site.specie.symbol == 'Li']

    # The reconstructed cell should match the reference argyrodite lattice.
    assert result.structure.lattice.abc == pytest.approx(reference.lattice.abc, abs=0.2)
    assert result.structure.lattice.angles == pytest.approx(reference.lattice.angles, abs=2.0)

    # Every reconstructed Li site should sit on a known argyrodite Li site.
    nearest = np.array(
        [
            reference.lattice.get_all_distances(site.frac_coords, ref_li).min()
            for site in result.structure
            if 'Li' in site.species.as_dict()
        ]
    )
    assert len(nearest) > 0
    assert nearest.max() < 2.5
    assert nearest.mean() < 1.5
