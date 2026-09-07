from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Species

from gemdat.density_crystallography import (
    crystallographic_density_metrics,
    fit_density_model,
    fold_supercell,
    periodic_gaussian_density,
    rank_spacegroups,
    symmetrize_density,
    trajectory_to_symmetrized_density,
    wyckoff_orbit,
)
from gemdat.trajectory import Trajectory

# Synthetic ground truth: a two-orbit Fm-3m (#225) Gaussian-mixture density on a
# small grid. Site A is the tetrahedral 8c orbit, site B the octahedral 4b
# orbit. Because each orbit is generated with ``wyckoff_orbit``, the density is
# exactly Fm-3m symmetric by construction.
GRID = 20
SIGMA_A = 0.05
SIGMA_B = 0.065
OCC_A = 0.7
POS_A = (0.25, 0.25, 0.25)
POS_B = (0.5, 0.5, 0.5)


def _synthetic_density(grid=GRID, sigma_a=SIGMA_A, sigma_b=SIGMA_B, occ_a=OCC_A, pos_a=POS_A):
    orbit_a = wyckoff_orbit(225, pos_a)
    orbit_b = wyckoff_orbit(225, POS_B)
    dens_a = periodic_gaussian_density(grid, orbit_a, sigma_a)
    dens_a /= dens_a.sum()
    dens_b = periodic_gaussian_density(grid, orbit_b, sigma_b)
    dens_b /= dens_b.sum()
    dens = occ_a * dens_a + (1.0 - occ_a) * dens_b
    return dens / dens.sum()


@pytest.fixture
def density():
    return _synthetic_density()


def test_fold_supercell_tiled():
    rng = np.random.default_rng(0)
    cell = rng.random((5, 7, 4))
    tiled = np.tile(cell, (2, 2, 2))

    folded = fold_supercell(tiled, (2, 2, 2))

    assert folded.shape == cell.shape
    assert np.allclose(folded, 8.0 * cell)
    # anisotropic supercell too
    assert np.allclose(fold_supercell(np.tile(cell, (3, 1, 2)), (3, 1, 2)), 6.0 * cell)


def test_fold_supercell_bad_divisor():
    with pytest.raises(ValueError):
        fold_supercell(np.zeros((5, 5, 5)), (2, 2, 2))


@pytest.mark.parametrize(
    'number,position,multiplicity',
    [
        (225, (0.0, 0.0, 0.0), 4),  # Fm-3m 4a
        (225, (0.25, 0.25, 0.25), 8),  # Fm-3m 8c
        (225, (0.5, 0.5, 0.5), 4),  # Fm-3m 4b
        (221, (0.0, 0.0, 0.0), 1),  # Pm-3m 1a
        (221, (0.0, 0.0, 0.5), 3),  # Pm-3m 3d
    ],
)
def test_wyckoff_orbit_multiplicity(number, position, multiplicity):
    orbit = wyckoff_orbit(number, position)
    assert orbit.shape == (multiplicity, 3)
    assert np.all((orbit >= 0.0) & (orbit < 1.0))


def test_symmetrize_density_idempotent(density):
    once = symmetrize_density(density, 225)
    twice = symmetrize_density(once, 225)

    # Synthetic density is already Fm-3m symmetric -> symmetrize is a near-identity
    assert np.allclose(density / density.max(), once / once.max(), atol=1e-6)
    assert np.allclose(once, twice, atol=1e-10)


def test_symmetrize_density_restores_symmetry(density):
    half = GRID // 2
    broken = density.copy()
    broken[:half, :half, :half] *= 1.6

    r1_broken = crystallographic_density_metrics(density, broken)['r1_like']
    r1_fixed = crystallographic_density_metrics(density, symmetrize_density(broken, 225))[
        'r1_like'
    ]

    assert r1_broken > 0.05
    assert r1_fixed < 0.1 * r1_broken


def test_crystallographic_density_metrics_identity(density):
    metrics = crystallographic_density_metrics(density, density)

    assert metrics['r1_like'] == pytest.approx(0.0, abs=1e-9)
    assert metrics['pearson_r'] == pytest.approx(1.0, abs=1e-9)
    assert metrics['jensen_shannon'] == pytest.approx(0.0, abs=1e-9)
    assert metrics['mae'] == pytest.approx(0.0, abs=1e-12)
    assert metrics['n_voxels'] == float(density.size)


def test_crystallographic_density_metrics_shuffled(density):
    shuffled = density.ravel().copy()
    np.random.default_rng(1).shuffle(shuffled)
    shuffled = shuffled.reshape(density.shape)

    identity = crystallographic_density_metrics(density, density)
    scrambled = crystallographic_density_metrics(density, shuffled)

    assert scrambled['r1_like'] > 20 * (identity['r1_like'] + 1e-9)
    assert scrambled['jensen_shannon'] > identity['jensen_shannon']
    assert scrambled['pearson_r'] < 0.5


def test_periodic_gaussian_density_single_position_shape():
    dens = periodic_gaussian_density(12, np.array([0.5, 0.5, 0.5]), 0.1)
    assert dens.shape == (12, 12, 12)
    # peak at the centre voxel
    assert np.unravel_index(np.argmax(dens), dens.shape) == (6, 6, 6)


def test_fit_density_model_recovers_planted_parameters(density):
    sites = [
        {'specie': 'Li', 'position': POS_A},
        {'specie': 'Li', 'position': POS_B},
    ]
    result = fit_density_model(
        density,
        225,
        sites,
        symmetrize=True,
        cell_length=10.0,
        maxiter=40,
        popsize=12,
        tol=1e-7,
        seed=0,
    )

    assert result.spacegroup_number == 225
    assert result.metrics['r1_like'] < 0.05

    tet, oct_ = result.sites
    assert tet.multiplicity == 8
    assert oct_.multiplicity == 4
    assert tet.sigma == pytest.approx(SIGMA_A, rel=0.05)
    assert oct_.sigma == pytest.approx(SIGMA_B, rel=0.05)
    assert tet.occupancy == pytest.approx(OCC_A, abs=0.03)
    assert tet.occupancy + oct_.occupancy == pytest.approx(1.0)
    assert tet.u_iso == pytest.approx((SIGMA_A * 10.0) ** 2, rel=0.1)


def test_fit_density_model_free_position_reports_split_scale():
    # Plant the tetrahedral site displaced off the ideal 1/4,1/4,1/4.
    dens = _synthetic_density(grid=16, sigma_a=0.045, pos_a=(0.28, 0.28, 0.28))
    sites = [
        {'specie': 'Li', 'position': POS_A, 'free_position': True, 'max_displacement': 0.08},
        {'specie': 'Li', 'position': POS_B},
    ]
    result = fit_density_model(
        dens,
        225,
        sites,
        symmetrize=True,
        cell_length=10.0,
        maxiter=15,
        popsize=8,
        tol=1e-6,
        seed=0,
    )

    assert result.metrics['r1_like'] < 0.15
    # the split length scale is reported in Angstrom and is clearly non-zero
    assert result.sites[0].displacement is not None
    assert result.sites[0].displacement > 0.1


def test_rank_spacegroups_puts_true_group_first(density):
    sites = [
        {'specie': 'Li', 'position': POS_A},
        {'specie': 'Li', 'position': POS_B},
    ]
    ranked = rank_spacegroups(
        density,
        candidates=[221, 225],
        sites_per_candidate=sites,
        maxiter=25,
        popsize=10,
        seed=0,
    )

    assert [r.spacegroup_number for r in ranked][0] == 225
    assert ranked[0].metrics['r1_like'] < ranked[1].metrics['r1_like']
    assert ranked[0].metrics['r1_like'] < 0.05


def test_trajectory_to_symmetrized_density_roundtrips():
    """A trajectory whose Li atoms sit on the Fm-3m 8c orbit should yield a
    symmetrised density that a matching single-site model fits well."""
    rng = np.random.default_rng(2)
    orbit = wyckoff_orbit(225, POS_A)  # 8 sites
    n_frames = 180

    coords = np.empty((n_frames, len(orbit) + 1, 3))
    for frame in range(n_frames):
        coords[frame, : len(orbit)] = orbit + rng.normal(scale=0.035, size=orbit.shape)
        coords[frame, len(orbit)] = np.array([0.0, 0.0, 0.0]) + rng.normal(scale=0.005, size=3)
    coords %= 1.0

    traj = Trajectory(
        species=[Species('Li')] * len(orbit) + [Species('S')],
        coords=coords,
        lattice=np.eye(3) * 8.0,
        time_step=1,
        metadata={'temperature': 300},
    )

    dens = trajectory_to_symmetrized_density(traj, 225, floating_specie='Li', resolution=0.5)

    assert dens.ndim == 3
    # symmetric under Fm-3m -> re-symmetrising changes almost nothing
    assert np.allclose(dens, symmetrize_density(dens, 225), atol=1e-9)

    result = fit_density_model(
        dens,
        225,
        [{'specie': 'Li', 'position': POS_A}],
        symmetrize=False,
        cell_length=8.0,
        maxiter=20,
        popsize=8,
        seed=0,
    )
    assert result.metrics['r1_like'] < 0.2
    assert result.sites[0].occupancy == pytest.approx(1.0)
