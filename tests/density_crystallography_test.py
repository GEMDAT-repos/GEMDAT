from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Lattice, Species, Structure

from gemdat.crystallizer import CrystallizerResult
from gemdat.density_crystallography import (
    crystallize_density_loop,
    crystallographic_density_metrics,
    fit_density_model,
    fold_supercell,
    from_crystallizer,
    periodic_gaussian_density,
    rank_spacegroups,
    site_free_directions,
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


@pytest.mark.parametrize(
    'number,position,directions',
    [
        (225, (0.25, 0.25, 0.25), []),  # Fm-3m 8c
        (225, (0.3, 0.3, 0.3), [[1, 1, 1]]),  # Fm-3m 32f (x,x,x)
        (225, (0.2, 0.0, 0.0), [[1, 0, 0]]),  # Fm-3m 24e (x,0,0)
        (225, (0.1, 0.1, 0.3), [[1, 1, 0], [0, 0, 1]]),  # Fm-3m 96k (x,x,z)
        (225, (0.11, 0.23, 0.37), np.eye(3)),  # general position
        (217, (0.0, 0.5, 0.5), []),  # I-43m 6b
        (191, (0.2, 0.4, 0.3), [[0.5, 1, 0], [0, 0, 1]]),  # P6/mmm 6l-like (x,2x,z)
    ],
)
def test_site_free_directions(number, position, directions):
    found = site_free_directions(number, position)

    assert found.shape == (len(directions), 3)
    assert np.allclose(found, np.reshape(directions, (-1, 3)))


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


def test_crystallographic_density_metrics_parameter_penalty(density):
    noisy = density * (1 + 0.1 * np.random.default_rng(3).standard_normal(density.shape))

    few = crystallographic_density_metrics(noisy, density, n_params=1)
    many = crystallographic_density_metrics(noisy, density, n_params=3)

    assert many['r1_like'] == few['r1_like']
    assert many['aic'] - few['aic'] == pytest.approx(4.0)
    assert many['bic'] - few['bic'] == pytest.approx(2 * np.log(density.size))
    assert many['gof'] > few['gof']


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
    dens = periodic_gaussian_density(11, np.array([0.5, 0.5, 0.5]), 0.1)
    assert dens.shape == (11, 11, 11)
    # voxel i is centred on (i + 1/2) / n, so 0.5 is the centre of voxel 5
    assert np.unravel_index(np.argmax(dens), dens.shape) == (5, 5, 5)


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


@pytest.mark.parametrize('planted', [0.0, 0.03])
def test_fit_density_model_split_site_displacement(planted):
    # The 8c site is really at (1/4 + d, 1/4 + d, 1/4 + d), i.e. split into 32f;
    # freeing only the (1, 1, 1) direction must report ~sqrt(3) * d * a, and ~0
    # when nothing is split.
    dens = _synthetic_density(grid=16, sigma_a=0.045, pos_a=(0.25 + planted,) * 3)
    sites = [
        {
            'specie': 'Li',
            'position': POS_A,
            'free_position': (1, 1, 1),
            'max_displacement': 0.08,
        },
        {'specie': 'Li', 'position': POS_B},
    ]
    result = fit_density_model(
        dens,
        225,
        sites,
        symmetrize=True,
        cell_length=10.0,
        maxiter=30,
        popsize=10,
        tol=1e-8,
        seed=0,
    )

    split = result.sites[0]
    assert split.multiplicity == 32
    assert split.displacement == pytest.approx(np.sqrt(3) * planted * 10.0, abs=0.1)
    assert split.sigma == pytest.approx(0.045, rel=0.05)
    assert result.sites[1].displacement is None


def test_fit_density_model_free_position_keeps_wyckoff_position():
    # 32f (x,x,x) at x = 0.28, started from x = 0.26: only x is refined
    dens = _synthetic_density(grid=16, sigma_a=0.045, pos_a=(0.28, 0.28, 0.28))
    sites = [
        {'specie': 'Li', 'position': (0.26, 0.26, 0.26), 'free_position': True},
        {'specie': 'Li', 'position': POS_B},
    ]
    result = fit_density_model(dens, 225, sites, maxiter=30, popsize=10, tol=1e-8, seed=0)

    site = result.sites[0]
    assert len(result.params) == 4  # 2 sigmas, x, 1 occupancy
    assert site.multiplicity == 32
    assert np.allclose(site.position, site.position[0])
    # x is 0.28 up to the Fm-3m equivalents 0.22, 0.72 and 0.78
    assert abs(site.position[0] % 0.5 - 0.25) == pytest.approx(0.03, abs=0.005)


def test_fit_density_model_coarsen(density):
    sites = [
        {'specie': 'Li', 'position': POS_A},
        {'specie': 'Li', 'position': POS_B},
    ]
    result = fit_density_model(density, 225, sites, maxiter=30, popsize=10, seed=0, coarsen=2)

    assert result.model_density.shape == density.shape
    assert result.metrics['r1_like'] < 0.05
    assert result.sites[0].sigma == pytest.approx(SIGMA_A, rel=0.05)
    assert result.sites[0].occupancy == pytest.approx(OCC_A, abs=0.03)

    with pytest.raises(ValueError, match='not divisible'):
        fit_density_model(density, 225, sites, maxiter=1, coarsen=3)


def test_fit_density_model_free_position_on_fixed_site_raises(density):
    sites = [{'specie': 'Li', 'position': POS_A, 'free_position': True}]
    with pytest.raises(ValueError, match='no free coordinates'):
        fit_density_model(density, 225, sites, maxiter=1)


def test_fit_density_model_workers(density):
    sites = [
        {'specie': 'Li', 'position': POS_A},
        {'specie': 'Li', 'position': POS_B},
    ]
    result = fit_density_model(density, 225, sites, maxiter=20, popsize=8, seed=0, workers=2)

    assert result.metrics['r1_like'] < 0.05
    assert result.sites[0].occupancy == pytest.approx(OCC_A, abs=0.03)


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
    assert ranked[0].ranking_metrics['bic'] < ranked[1].ranking_metrics['bic']
    assert ranked[0].metrics['r1_like'] < 0.05


def test_rank_spacegroups_bad_criterion(density):
    with pytest.raises(ValueError, match='criterion'):
        rank_spacegroups(density, [225], [{'specie': 'Li', 'position': POS_A}], criterion='x')


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
    # to_volume voxels are registered correctly: sigma is the thermal width
    # plus the histogram bin width, not doubled by a half-voxel offset
    h = 1.0 / dens.shape[0]
    assert result.sites[0].sigma == pytest.approx(np.sqrt(0.035**2 + h**2 / 12), rel=0.1)


def _antifluorite_supercell(shift=(0.0, 0.0, 0.0), noise=0.0, drop=None):
    """2x2x2 supercell of Fm-3m S 4a + Li 8c, as a crystallizer result."""
    structure = Structure.from_spacegroup(
        225, Lattice.cubic(7.0), ['S', 'Li'], [[0, 0, 0], [0.25, 0.25, 0.25]]
    )
    structure.make_supercell(2)
    if drop is not None:
        structure.remove_sites([drop])
    rng = np.random.default_rng(0)
    cart = structure.cart_coords + rng.normal(scale=noise, size=(len(structure), 3))
    structure = Structure(structure.lattice, structure.species, cart, coords_are_cartesian=True)
    structure.translate_sites(range(len(structure)), shift)
    return CrystallizerResult(structure, 'Fm-3m', 225, symprec=0.3)


def test_from_crystallizer_folds_noisy_shifted_supercell():
    shift = np.array([0.013, 0.021, -0.007])  # in supercell fractions
    ops, sites = from_crystallizer(_antifluorite_supercell(shift, noise=0.05), supercell=2)

    assert len(ops) == 192
    assert [site['specie'] for site in sites] == ['S', 'Li']
    # positions sit exactly on their site symmetry, so the orbits close
    assert [len(wyckoff_orbit(ops, site['position'])) for site in sites] == [4, 8]
    # operations fitted to MD data are exact only to ~1e-5, and orbits of
    # positions typed by hand at that precision must still close
    assert len(wyckoff_orbit(ops, sites[0]['position'] + 0.5 + 1e-6)) == 4
    # the folded origin follows the structure, without a standard setting
    offset = sites[0]['position'] - 2 * shift
    assert np.allclose(offset - np.round(offset), 0, atol=0.01)


def test_from_crystallizer_ops_match_standard_setting(density):
    ops, sites = from_crystallizer(_antifluorite_supercell(), supercell=2)

    assert np.allclose(symmetrize_density(density, ops), symmetrize_density(density, 225))
    result = fit_density_model(
        density, ops, [s for s in sites if s['specie'] == 'Li'], maxiter=5, popsize=5, seed=0
    )
    assert result.spacegroup_symbol == 'Fm-3m'
    assert result.sites[0].multiplicity == 8


def test_from_crystallizer_without_supercell_repeat_raises():
    result = _antifluorite_supercell(drop=0)  # one S vacancy breaks the repeat
    result.spacegroup_symbol, result.spacegroup_number = 'Pm-3m', 221

    with pytest.raises(ValueError, match='cannot be folded'):
        from_crystallizer(result, supercell=2)
    ops, _ = from_crystallizer(result)
    assert len(ops) == 48


def test_crystallize_density_loop_finds_the_site_hidden_under_the_peaks():
    """A sharp 8c orbit plus a broad, weak 4b orbit: the peak finder sees only
    8c, so the loop has to recover 4b from what the first fit leaves."""
    rng = np.random.default_rng(0)
    tetrahedral = wyckoff_orbit(225, POS_A)
    octahedral = wyckoff_orbit(225, POS_B)
    framework = wyckoff_orbit(225, (0.0, 0.0, 0.0))
    n_frames = 600
    n_li = len(tetrahedral) + len(octahedral)

    coords = np.empty((n_frames, n_li + len(framework), 3))
    for frame in range(n_frames):
        coords[frame, : len(tetrahedral)] = tetrahedral + rng.normal(
            0, 0.035, tetrahedral.shape
        )
        coords[frame, len(tetrahedral) : n_li] = octahedral + rng.normal(
            0, 0.07, octahedral.shape
        )
        coords[frame, n_li:] = framework + rng.normal(0, 0.008, framework.shape)
    coords %= 1.0

    traj = Trajectory(
        species=[Species('Li')] * n_li + [Species('S')] * len(framework),
        coords=coords,
        lattice=np.eye(3) * 8.0,
        time_step=1,
        metadata={'temperature': 300},
    )

    rounds = crystallize_density_loop(
        traj,
        'Li',
        framework_species=['S'],
        resolution=0.3,
        background_level=0.3,
        max_rounds=3,
        cell_length=8.0,
        maxiter=30,
        popsize=8,
        seed=0,
    )['Li']

    assert len(rounds) >= 2
    first, second = rounds[0], rounds[1]

    # Round 0 sees the 8 tetrahedral peaks and nothing else.
    assert first.fit.spacegroup_symbol == 'Fm-3m'
    assert [site.multiplicity for site in first.fit.sites] == [8]
    assert 0.3 < first.explained < 0.9  # the octahedral density is left over

    # Round 1 recovers the octahedral orbit from that leftover, with the width
    # it was planted at, and the density is then used up.
    octahedral_fit = next(site for site in second.fit.sites if site.multiplicity == 4)
    half_integer = octahedral_fit.position * 2
    assert np.allclose(half_integer - np.round(half_integer), 0, atol=0.02)
    assert octahedral_fit.sigma == pytest.approx(0.07, abs=0.02)
    assert first.explained + second.explained > 0.95
    assert second.fit.metrics['r1_like'] < first.fit.metrics['r1_like']

    # Densities never go negative, and every round shrinks what is left.
    assert all((round_.residual >= 0).all() for round_ in rounds)
    assert second.residual.sum() < first.residual.sum()
