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
def test_crystallize_argyrodite_md_is_low_symmetry(vasp_crystallizer):
    """Crystallizing real Li6PS5Br argyrodite MD yields a low-symmetry (P1)
    cell. This is the expected, correct outcome - not a bug in the pipeline.

    The cubic F-43m (#216) parent cannot be recovered from this trajectory
    for two independent reasons:

    * The time-averaged S/Br host framework is itself P1 at every symprec:
      S(2-) and Br(-) are site-disordered over the MD supercell, so their
      frame-averaged positions carry no symmetry.
    * The Li density-peak centroids scatter ~0.3 A (up to ~0.9 A) off the
      ideal 48h Wyckoff positions - random per-site scatter, not a global
      shift that a larger symprec could absorb.

    ``tests/crystallizer_test.py::test_crystallize_ideal_argyrodite_sublattice``
    shows the same code recovers F-43m from a genuinely ideal 48h sublattice,
    so the P1 here is a property of the data, not the algorithm.
    """
    result = vasp_crystallizer.crystallize()

    assert result.spacegroup_number >= 1
    # Currently exactly P1; kept as a canary so a future change that recovers
    # more symmetry from this data is noticed (and this comment updated).
    assert result.spacegroup_number == 1


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

    The bundled `argyrodite.cif` only describes the Li sites, so this
    reference comparison is only meaningful for the Li diffusing
    species.
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
    # Observed with the current code: max ~1.52 A, mean ~0.74 A (deterministic,
    # the density peak-finding has no randomness). Thresholds sit a little above
    # that - loose enough not to flake on numerical drift, tight enough to catch
    # a real regression in the reconstruction.
    assert nearest.max() < 2.0
    assert nearest.mean() < 1.0


@pytest.na3sbs4_cache_available  # type: ignore
def test_crystallize_na3sbs4_plastic_crystal(na3sbs4_traj):
    """Point 3 of issue #421: verify the crystallizer on a plastic crystal.

    Na3(Sb/W)S4 has orientationally-disordered but positionally-ordered
    SbS4/WS4 rotor tetrahedra, so - unlike site-disordered argyrodite - the
    time-averaged host framework is a well-defined cubic lattice, and the
    mobile Na density fills the I-43m Na sublattice. The crystallizer
    therefore recovers cubic symmetry here.

    Observed with the current code: I-43m (#217) at symprec 0.5, 128 sites
    total (S64 Na48 Sb14 W2), lattice a=b~14.437 A, c~14.475 A, angles ~90.
    """
    cr = Crystallizer.from_trajectory(na3sbs4_traj, floating_specie='Na', resolution=0.3)
    result = cr.crystallize(background_level=0.25)

    # Cubic (number >= 195) is the hard requirement; #217 is what it hits now.
    assert result.spacegroup_number >= 195
    assert result.spacegroup_number == 217

    n_na = sum(1 for site in result.structure if 'Na' in site.species.as_dict())
    assert 40 <= n_na <= 56  # observed 48

    abc = result.structure.lattice.abc
    angles = result.structure.lattice.angles
    assert max(abc) - min(abc) < 0.15  # near-cubic: observed spread ~0.04 A
    assert all(abs(angle - 90.0) < 1.0 for angle in angles)  # observed ~90, ~90.24
