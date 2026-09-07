"""Density-space crystallography: derive an average-structure model by fitting
a symmetry-constrained Gaussian mixture *directly* to a species-resolved MD
nuclear probability density.

This is a generalisation of the "MD crystallography" reference workflow developed
by T. Famprikis for plastic-crystal alpha-Li2SO4 (space group Fm-3m). The reference
hard-codes the Fm-3m symmetry operations; here every symmetry operation (rotations,
translations and centering) is taken from a user-supplied space group via
:mod:`pymatgen.symmetry.groups`, so the same machinery works for an arbitrary
space group.

The pipeline is:

1. Build a species-resolved density on a grid (``Trajectory.to_volume``).
2. Optionally fold a supercell density down to the primitive/conventional cell
   (:func:`fold_supercell`).
3. Symmetrise the density with the full operation set of a candidate space group
   (:func:`symmetrize_density`).
4. Fit a Gaussian-mixture average-structure model (one isotropic Gaussian per
   Wyckoff orbit, mixed by occupancy fraction) against the symmetrised density
   (:func:`fit_density_model`), refining the Gaussian width (``sigma`` ->
   ``U_iso``), the occupancy fractions and, optionally, free positional
   parameters (the displacement scale at which a higher-symmetry site splits).
5. Compare candidate space groups on the fit residual (:func:`rank_spacegroups`).

This addresses GitHub issue #421 points 1 (assume / rank the space group),
4 (incorporate density -- fit the density directly) and 5 (Wyckoff subsymmetry
and the length scale in Angstrom at which sites split).

Deferred follow-ups (not in this module yet): the anisotropic-ADP O-style fit,
the per-species ordered S -> Li -> O workflow with origin-shift propagation, and
plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from pymatgen.symmetry.groups import SpaceGroup
from scipy.ndimage import map_coordinates
from scipy.optimize import differential_evolution

if TYPE_CHECKING:
    from .trajectory import Trajectory

__all__ = [
    'DensityFitResult',
    'SiteFit',
    'crystallographic_density_metrics',
    'fit_density_model',
    'fold_supercell',
    'periodic_anisotropic_gaussian_density',
    'periodic_gaussian_density',
    'rank_spacegroups',
    'symmetrize_density',
    'trajectory_to_symmetrized_density',
    'wyckoff_orbit',
]

_EPS = 1e-12

Shape = tuple[int, int, int]


def _as_spacegroup(spacegroup: str | int | SpaceGroup) -> SpaceGroup:
    """Coerce a space-group symbol, international number or object to a
    :class:`pymatgen.symmetry.groups.SpaceGroup`."""
    if isinstance(spacegroup, SpaceGroup):
        return spacegroup
    if isinstance(spacegroup, (int, np.integer)):
        return SpaceGroup.from_int_number(int(spacegroup))
    return SpaceGroup(str(spacegroup))


_OP_ARRAY_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _op_arrays(sg: SpaceGroup) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(rotations, translations)`` stacks for every symmetry operation
    of ``sg`` -- ``rotations`` is ``(K, 3, 3)`` and ``translations`` is ``(K,
    3)`` -- cached per international number so the pymatgen operation set is
    only iterated once per space group."""
    key = int(sg.int_number)
    cached = _OP_ARRAY_CACHE.get(key)
    if cached is None:
        ops = list(sg.symmetry_ops)
        rotations = np.array([np.asarray(op.rotation_matrix, dtype=float) for op in ops])
        translations = np.array([np.asarray(op.translation_vector, dtype=float) for op in ops])
        cached = (rotations, translations)
        _OP_ARRAY_CACHE[key] = cached
    return cached


def _as_shape(grid_size: int | tuple[int, int, int]) -> Shape:
    """Normalise ``grid_size`` to a ``(nx, ny, nz)`` tuple."""
    if isinstance(grid_size, (int, np.integer)):
        return (int(grid_size), int(grid_size), int(grid_size))
    a, b, c = grid_size
    return (int(a), int(b), int(c))


def _fractional_grid(shape: Shape) -> np.ndarray:
    """Return the ``(N, 3)`` array of fractional coordinates of every voxel
    centre of a grid with the given ``shape`` (voxel ``i`` sits at ``i /
    n``)."""
    axes = [np.arange(n, dtype=float) / n for n in shape]
    grids = np.meshgrid(*axes, indexing='ij')
    return np.stack([g.ravel() for g in grids], axis=-1)


def fold_supercell(grid: np.ndarray, supercell: tuple[int, int, int]) -> np.ndarray:
    """Fold a supercell density onto a single cell by summing the blocks.

    Generalises the reference ``fold_pdf`` (which only handled a 2x2x2 cubic
    supercell) to an arbitrary integer supercell.

    Parameters
    ----------
    grid : np.ndarray
        3D density on a grid whose dimensions are integer multiples of
        ``supercell``.
    supercell : tuple[int, int, int]
        Number of cells along each axis.

    Returns
    -------
    np.ndarray
        Folded density with shape ``grid.shape // supercell``. The sum of the
        folded density equals the sum of the input.
    """
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 3:
        raise ValueError('grid must be a 3D array')

    sc = _as_shape(supercell)
    if any(s < 1 for s in sc):
        raise ValueError('supercell entries must be >= 1')

    for n, s, axis in zip(grid.shape, sc, 'xyz'):
        if n % s != 0:
            raise ValueError(
                f'grid dimension {axis}={n} is not divisible by supercell factor {s}'
            )

    mx, my, mz = (n // s for n, s in zip(grid.shape, sc))
    folded = np.zeros((mx, my, mz), dtype=float)
    for i in range(sc[0]):
        for j in range(sc[1]):
            for k in range(sc[2]):
                folded += grid[
                    i * mx : (i + 1) * mx,
                    j * my : (j + 1) * my,
                    k * mz : (k + 1) * mz,
                ]
    return folded


def symmetrize_density(
    grid: np.ndarray,
    spacegroup: str | int | SpaceGroup,
    *,
    order: int = 1,
) -> np.ndarray:
    """Symmetrise a density grid with the full operation set of a space group.

    Every symmetry operation of ``spacegroup`` (proper/improper rotations,
    screw/glide translations *and* centering translations, as returned by
    pymatgen) is applied to the grid by resampling with
    :func:`scipy.ndimage.map_coordinates` under periodic (``grid-wrap``)
    boundaries, and the results are averaged. This replaces both
    ``fold_fcc_translations`` and ``symmetrize`` of the reference with one
    space-group-driven routine.

    Parameters
    ----------
    grid : np.ndarray
        3D density on a (not necessarily cubic) grid, indexed so that voxel
        ``i`` corresponds to fractional coordinate ``i / n``.
    spacegroup : str | int | SpaceGroup
        International symbol (``'Fm-3m'``), international number (``225``) or a
        :class:`pymatgen.symmetry.groups.SpaceGroup`.
    order : int, optional
        Spline order for :func:`scipy.ndimage.map_coordinates` (1 = trilinear).

    Returns
    -------
    np.ndarray
        Symmetrised density, same shape as ``grid``.
    """
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 3:
        raise ValueError('grid must be a 3D array')

    sg = _as_spacegroup(spacegroup)
    shape: Shape = (grid.shape[0], grid.shape[1], grid.shape[2])
    coords = _fractional_grid(shape)
    n = np.array(shape, dtype=float)

    rotations, translations = _op_arrays(sg)
    acc = np.zeros(grid.size, dtype=np.float64)
    for rot, trans in zip(rotations, translations):
        transformed = (coords @ rot.T + trans) % 1.0
        sample = (transformed * n).T
        acc += map_coordinates(grid, sample, order=order, mode='grid-wrap')

    return (acc / len(rotations)).reshape(shape)


def wyckoff_orbit(
    spacegroup: str | int | SpaceGroup,
    position: np.ndarray,
    *,
    tol: float = 1e-8,
) -> np.ndarray:
    """Full symmetry orbit of a representative fractional coordinate.

    Generalises the reference ``generate_fm3m_orbit`` /
    ``unique_periodic_positions``: applies every operation of ``spacegroup`` to
    ``position`` and deduplicates the images under periodic boundaries. The
    number of returned points is the site multiplicity in the setting of
    ``spacegroup``.

    Parameters
    ----------
    spacegroup : str | int | SpaceGroup
        Space group (symbol, number or object).
    position : np.ndarray
        Representative fractional coordinate, shape ``(3,)``.
    tol : float, optional
        Distance below which two images (under the minimum-image convention)
        are considered identical.

    Returns
    -------
    np.ndarray
        Array of shape ``(multiplicity, 3)`` with the unique orbit positions in
        ``[0, 1)``.
    """
    sg = _as_spacegroup(spacegroup)
    rotations, translations = _op_arrays(sg)
    pos = np.asarray(position, dtype=float) % 1.0

    images = np.mod(np.einsum('kij,j->ki', rotations, pos) + translations, 1.0)

    # Deduplicate under periodic boundaries. Distinct orbit points are always
    # well separated, so rounding to the tolerance's decimal place (after
    # wrapping ~1.0 back to 0.0) is a safe, vectorised key.
    decimals = max(1, int(round(-np.log10(tol))))
    keyed = np.mod(np.round(images, decimals), 1.0)
    _, keep = np.unique(keyed, axis=0, return_index=True)
    return images[np.sort(keep)]


def periodic_gaussian_density(
    grid_size: int | tuple[int, int, int],
    positions: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Periodic isotropic Gaussian density summed over a set of positions.

    Faithful port of the reference ``periodic_gaussian_density`` (minimum-image
    convention on a fractional grid), accepting a non-cubic ``grid_size``.

    Parameters
    ----------
    grid_size : int | tuple[int, int, int]
        Grid shape (a scalar means a cubic grid).
    positions : np.ndarray
        Fractional coordinates, shape ``(M, 3)`` (a single ``(3,)`` is allowed).
    sigma : float
        Isotropic Gaussian width in fractional-coordinate units.

    Returns
    -------
    np.ndarray
        Unnormalised density with shape ``grid_size``.
    """
    shape = _as_shape(grid_size)
    axes = [np.linspace(0.0, 1.0, n, endpoint=False) for n in shape]
    grid_x, grid_y, grid_z = np.meshgrid(*axes, indexing='ij')

    pts = np.atleast_2d(np.asarray(positions, dtype=float))
    two_sigma_sq = 2.0 * float(sigma) ** 2
    density = np.zeros(shape, dtype=float)
    for px, py, pz in pts:
        dx = grid_x - px
        dx -= np.round(dx)
        dy = grid_y - py
        dy -= np.round(dy)
        dz = grid_z - pz
        dz -= np.round(dz)
        density += np.exp(-(dx * dx + dy * dy + dz * dz) / two_sigma_sq)
    return density


def periodic_anisotropic_gaussian_density(
    grid_size: int | tuple[int, int, int],
    positions: np.ndarray,
    cov_matrices: np.ndarray,
) -> np.ndarray:
    """Periodic anisotropic Gaussian density summed over a set of positions.

    Experimental / lightly tested port of the reference
    ``_single_anisotropic_gaussian_grid`` summed over an orbit. Intended as a
    building block for a future anisotropic-ADP fit; :func:`fit_density_model`
    does not use it yet.

    Parameters
    ----------
    grid_size : int | tuple[int, int, int]
        Grid shape.
    positions : np.ndarray
        Fractional coordinates, shape ``(M, 3)``.
    cov_matrices : np.ndarray
        Either one ``(3, 3)`` fractional covariance (ADP) tensor applied to
        every position, or a per-position stack of shape ``(M, 3, 3)``.

    Returns
    -------
    np.ndarray
        Unnormalised density with shape ``grid_size``.
    """
    shape = _as_shape(grid_size)
    axes = [np.linspace(0.0, 1.0, n, endpoint=False) for n in shape]
    grid_x, grid_y, grid_z = np.meshgrid(*axes, indexing='ij')
    flat = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)

    pts = np.atleast_2d(np.asarray(positions, dtype=float))
    cov = np.asarray(cov_matrices, dtype=float)
    if cov.ndim == 2:
        cov = np.broadcast_to(cov, (len(pts), 3, 3))

    density = np.zeros(len(flat), dtype=float)
    for pos, cmat in zip(pts, cov):
        delta = (flat - pos + 0.5) % 1.0 - 0.5
        cinv = np.linalg.inv(cmat)
        mahalanobis = np.einsum('ni,ij,nj->n', delta, cinv, delta)
        density += np.exp(-0.5 * mahalanobis)
    return density.reshape(shape)


def crystallographic_density_metrics(
    observed: np.ndarray,
    calculated: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
    n_params: int = 0,
    eps: float = _EPS,
) -> dict[str, float]:
    """Crystallography-flavoured goodness-of-fit metrics for two 3D densities.

    Faithful port of the reference ``crystallographic_pdf_metrics`` with
    snake_case keys. Both grids are independently renormalised to sum to 1
    before comparison, so a raw (unnormalised) observed density and an
    already-normalised model density can both be passed directly.

    Parameters
    ----------
    observed, calculated : np.ndarray
        Same-shaped 3D arrays.
    mask : np.ndarray | None, optional
        Boolean array selecting voxels to include (e.g. to exclude background).
    sigma : np.ndarray | None, optional
        Pointwise standard deviations for a weighted residual (weights
        ``w = 1 / sigma**2``). Must match the full grid shape.
    n_params : int, optional
        Number of fitted parameters, used only for the goodness-of-fit ``gof``.
    eps : float, optional
        Small stabiliser to avoid division by zero.

    Returns
    -------
    dict[str, float]
        Keys: ``r1_like``, ``wr_like``, ``gof``, ``mse``, ``mae``, ``rmse``,
        ``pearson_r``, ``jensen_shannon``, ``n_voxels``.
    """
    obs = np.asarray(observed, dtype=float)
    calc = np.asarray(calculated, dtype=float)
    if obs.shape != calc.shape:
        raise ValueError('observed and calculated must have the same shape')

    obs = obs / (np.sum(obs) + eps)
    calc = calc / (np.sum(calc) + eps)

    if mask is None:
        sel = np.ones(obs.shape, dtype=bool)
    else:
        sel = np.asarray(mask, dtype=bool)
        if sel.shape != obs.shape:
            raise ValueError('mask must have the same shape as the densities')

    if sigma is None:
        weights_full = np.ones(obs.shape, dtype=float)
    else:
        sig = np.asarray(sigma, dtype=float)
        if sig.shape != obs.shape:
            raise ValueError('sigma must have the same shape as the densities')
        weights_full = 1.0 / (sig**2 + eps)

    obs_m = obs[sel]
    calc_m = calc[sel]
    weights = weights_full[sel]
    if obs_m.size == 0:
        raise ValueError('mask selected no voxels')

    residual = obs_m - calc_m

    r1_like = float(np.sum(np.abs(residual)) / (np.sum(np.abs(obs_m)) + eps))
    wr_like = float(np.sqrt(np.sum(weights * residual**2) / (np.sum(weights * obs_m**2) + eps)))
    dof = max(obs_m.size - int(n_params), 1)
    gof = float(np.sqrt(np.sum(weights * residual**2) / dof))

    mse = float(np.mean(residual**2))
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(mse))
    pearson_r = float(np.corrcoef(obs_m, calc_m)[0, 1]) if obs_m.size > 1 else float('nan')

    p = obs_m.ravel() + eps
    q = calc_m.ravel() + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    jensen_shannon = float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))

    return {
        'r1_like': r1_like,
        'wr_like': wr_like,
        'gof': gof,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'pearson_r': pearson_r,
        'jensen_shannon': jensen_shannon,
        'n_voxels': float(obs_m.size),
    }


@dataclass
class SiteFit:
    """Refined parameters of a single Wyckoff-orbit site.

    Parameters
    ----------
    specie : str
        Chemical species label.
    position : np.ndarray
        Refined representative fractional coordinate.
    initial_position : np.ndarray
        Representative fractional coordinate the fit started from.
    multiplicity : int
        Number of symmetry-equivalent sites in the orbit.
    sigma : float
        Refined isotropic Gaussian width (fractional-coordinate units).
    occupancy : float
        Integrated fraction of the total model density carried by this orbit.
    u_iso : float | None
        Isotropic displacement parameter ``(sigma * a)**2`` in Angstrom^2, or
        ``None`` when no cell edge was supplied. Assumes a (near-)cubic cell.
    displacement : float | None
        ``|position - initial_position|`` in Angstrom -- the length scale at
        which the site is displaced from its higher-symmetry ideal position
        (issue #421 point 5). ``None`` for a fixed-position site or when no
        cell edge was supplied.
    """

    specie: str
    position: np.ndarray
    initial_position: np.ndarray
    multiplicity: int
    sigma: float
    occupancy: float
    u_iso: float | None = None
    displacement: float | None = None


@dataclass
class DensityFitResult:
    """Result of :func:`fit_density_model`.

    Parameters
    ----------
    spacegroup_symbol : str
        International symbol of the space group used.
    spacegroup_number : int
        International number of the space group used.
    sites : list[SiteFit]
        Refined per-orbit parameters, in input order.
    observed_density : np.ndarray
        The (folded and) symmetrised observed density the model was fit to.
    model_density : np.ndarray
        The best-fit model density (normalised to sum to 1).
    metrics : dict[str, float]
        Output of :func:`crystallographic_density_metrics` for the fit.
    loss : float
        Final optimiser objective (mean squared error of the normalised grids).
    success : bool
        Optimiser success flag.
    params : np.ndarray
        Raw refined parameter vector.
    """

    spacegroup_symbol: str
    spacegroup_number: int
    sites: list[SiteFit]
    observed_density: np.ndarray
    model_density: np.ndarray
    metrics: dict[str, float]
    loss: float
    success: bool
    params: np.ndarray = field(repr=False)


def _stick_breaking(raw: list[float]) -> np.ndarray:
    """Map ``n - 1`` numbers in ``(0, 1)`` to ``n`` non-negative fractions that
    sum to 1 (stick-breaking). For ``n == 2`` this reduces to
    ``[r, 1 - r]``, matching the reference's single ``alpha`` parameter."""
    fractions: list[float] = []
    remaining = 1.0
    for value in raw:
        take = remaining * float(value)
        fractions.append(take)
        remaining -= take
    fractions.append(remaining)
    return np.asarray(fractions, dtype=float)


def _prepare_observed(
    observed_density: np.ndarray,
    sg: SpaceGroup,
    *,
    symmetrize: bool,
    supercell: tuple[int, int, int] | None,
) -> np.ndarray:
    """Fold + symmetrise the observed density as requested."""
    obs = np.asarray(observed_density, dtype=float)
    if supercell is not None and _as_shape(supercell) != (1, 1, 1):
        obs = fold_supercell(obs, supercell)
    if symmetrize:
        obs = symmetrize_density(obs, sg)
    return obs


def fit_density_model(
    observed_density: np.ndarray,
    spacegroup: str | int | SpaceGroup,
    sites: list[dict[str, Any]],
    *,
    symmetrize: bool = True,
    supercell: tuple[int, int, int] | None = None,
    cell_length: float | None = None,
    sigma_bounds: tuple[float, float] = (0.01, 0.25),
    max_displacement: float = 0.15,
    maxiter: int = 100,
    popsize: int = 15,
    tol: float = 1e-4,
    seed: int | None = None,
    polish: bool = True,
    workers: int = 1,
) -> DensityFitResult:
    """Fit a symmetry-constrained isotropic Gaussian-mixture average structure.

    Each entry of ``sites`` contributes one isotropic Gaussian *per symmetry
    orbit* of a representative fractional coordinate; the orbits are mixed by
    refined occupancy fractions. Refined parameters are, per site, the Gaussian
    width ``sigma`` (and, optionally, the three fractional coordinates of the
    representative), plus ``n_sites - 1`` inter-site occupancy fractions.

    Parameters
    ----------
    observed_density : np.ndarray
        3D density, e.g. ``Trajectory.filter('Li').to_volume(...).data`` or the
        output of :func:`trajectory_to_symmetrized_density`.
    spacegroup : str | int | SpaceGroup
        Candidate space group.
    sites : list[dict]
        One spec per crystallographic site, e.g.
        ``{'specie': 'Li', 'position': (0.25, 0.25, 0.25), 'kind': 'isotropic',
        'free_position': False}``. Optional per-site keys: ``sigma_bounds``,
        ``max_displacement``. Only ``kind='isotropic'`` is currently supported.
    symmetrize : bool, optional
        Symmetrise ``observed_density`` with ``spacegroup`` before fitting.
    supercell : tuple[int, int, int] | None, optional
        If given, fold ``observed_density`` by this factor first.
    cell_length : float | None, optional
        Conventional (unit) cell edge in Angstrom, used to report ``u_iso`` and
        the split ``displacement``. Assumes a (near-)cubic cell.
    sigma_bounds : tuple[float, float], optional
        Global bounds for every ``sigma`` (fractional units).
    max_displacement : float, optional
        Global +/- bound (fractional units) for free positional parameters.
    maxiter, popsize, tol, seed, polish, workers
        Passed to :func:`scipy.optimize.differential_evolution`.

    Returns
    -------
    DensityFitResult
    """
    sg = _as_spacegroup(spacegroup)
    obs = _prepare_observed(observed_density, sg, symmetrize=symmetrize, supercell=supercell)
    if obs.ndim != 3:
        raise ValueError('observed_density must be a 3D array')
    shape: Shape = (obs.shape[0], obs.shape[1], obs.shape[2])
    obs_norm = obs / (np.sum(obs) + _EPS)

    specs: list[dict[str, Any]] = []
    for raw in sites:
        kind = str(raw.get('kind', 'isotropic'))
        if kind != 'isotropic':
            raise NotImplementedError(
                "only kind='isotropic' is supported; anisotropic-ADP fitting is a "
                'planned follow-up'
            )
        specs.append(
            {
                'specie': str(raw['specie']),
                'position': np.asarray(raw['position'], dtype=float) % 1.0,
                'free_position': bool(raw.get('free_position', False)),
                'sigma_bounds': tuple(raw.get('sigma_bounds', sigma_bounds)),
                'max_displacement': float(raw.get('max_displacement', max_displacement)),
            }
        )

    n_sites = len(specs)
    if n_sites == 0:
        raise ValueError('sites must contain at least one site spec')

    # Precompute orbits for fixed-position sites (cheap, but avoids recomputing
    # the deduplicated orbit on every objective evaluation).
    orbit_cache: dict[int, np.ndarray] = {}
    for idx, spec in enumerate(specs):
        if not spec['free_position']:
            orbit_cache[idx] = wyckoff_orbit(sg, spec['position'])

    bounds: list[tuple[float, float]] = [spec['sigma_bounds'] for spec in specs]
    for spec in specs:
        if spec['free_position']:
            p0 = spec['position']
            md = spec['max_displacement']
            bounds.extend((float(p0[c] - md), float(p0[c] + md)) for c in range(3))
    bounds.extend((1e-3, 1.0 - 1e-3) for _ in range(n_sites - 1))

    def unpack(params: np.ndarray) -> tuple[list[float], list[np.ndarray], np.ndarray]:
        cursor = 0
        sigmas = [float(v) for v in params[cursor : cursor + n_sites]]
        cursor += n_sites
        positions: list[np.ndarray] = []
        for spec in specs:
            if spec['free_position']:
                positions.append(np.asarray(params[cursor : cursor + 3], dtype=float))
                cursor += 3
            else:
                positions.append(spec['position'])
        raw_fracs = [float(v) for v in params[cursor : cursor + (n_sites - 1)]]
        return sigmas, positions, _stick_breaking(raw_fracs)

    def model(params: np.ndarray) -> np.ndarray | None:
        sigmas, positions, alphas = unpack(params)
        total = np.zeros(shape, dtype=float)
        for idx, (sigma, pos, alpha) in enumerate(zip(sigmas, positions, alphas)):
            if sigma <= 0:
                return None
            orbit = orbit_cache.get(idx)
            if orbit is None:
                orbit = wyckoff_orbit(sg, pos)
            single = periodic_gaussian_density(shape, orbit, sigma)
            norm = single.sum()
            if norm <= 0:
                return None
            total += alpha * (single / norm)
        grand = total.sum()
        if grand <= 0:
            return None
        return total / grand

    def loss(params: np.ndarray) -> float:
        grid = model(params)
        if grid is None:
            return np.inf
        return float(np.mean((grid - obs_norm) ** 2))

    result = differential_evolution(
        loss,
        bounds,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        seed=seed,
        polish=polish,
        workers=workers,
        updating='deferred' if workers != 1 else 'immediate',
    )

    sigmas, positions, alphas = unpack(result.x)
    model_density = model(result.x)
    assert model_density is not None
    metrics = crystallographic_density_metrics(obs, model_density, n_params=len(result.x))

    site_fits: list[SiteFit] = []
    for idx, (spec, sigma, pos, alpha) in enumerate(zip(specs, sigmas, positions, alphas)):
        orbit = orbit_cache.get(idx)
        if orbit is None:
            orbit = wyckoff_orbit(sg, pos)
        u_iso: float | None = None
        displacement: float | None = None
        if cell_length is not None:
            u_iso = float((sigma * cell_length) ** 2)
            delta = np.asarray(pos, dtype=float) - spec['position']
            delta -= np.round(delta)
            displacement = float(np.linalg.norm(delta) * cell_length)
        site_fits.append(
            SiteFit(
                specie=spec['specie'],
                position=np.asarray(pos, dtype=float) % 1.0,
                initial_position=np.asarray(spec['position'], dtype=float),
                multiplicity=len(orbit),
                sigma=float(sigma),
                occupancy=float(alpha),
                u_iso=u_iso,
                displacement=displacement,
            )
        )

    return DensityFitResult(
        spacegroup_symbol=sg.symbol,
        spacegroup_number=sg.int_number,
        sites=site_fits,
        observed_density=obs,
        model_density=model_density,
        metrics=metrics,
        loss=float(result.fun),
        success=bool(result.success),
        params=np.asarray(result.x, dtype=float),
    )


def rank_spacegroups(
    observed_density: np.ndarray,
    candidates: list[str | int | SpaceGroup],
    sites_per_candidate: list[dict[str, Any]] | list[list[dict[str, Any]]],
    *,
    symmetrize: bool = True,
    supercell: tuple[int, int, int] | None = None,
    cell_length: float | None = None,
    maxiter: int = 25,
    popsize: int = 10,
    seed: int | None = 0,
    **fit_kwargs: Any,
) -> list[DensityFitResult]:
    """Rank candidate space groups by how well a symmetry-constrained model
    fits.

    This is the density-fit analogue of "assume / rank the space group"
    (issue #421 point 1): for each candidate, symmetrise the observed density
    with that group and run a (deliberately cheap) :func:`fit_density_model`,
    then sort the results by ``r1_like`` (best first).

    Parameters
    ----------
    observed_density : np.ndarray
        3D density.
    candidates : list
        Candidate space groups (symbols, numbers or objects).
    sites_per_candidate : list
        Either a single list of site specs (used for every candidate) or a list
        of such lists, one per candidate.
    symmetrize, supercell, cell_length
        Forwarded to :func:`fit_density_model`.
    maxiter, popsize, seed
        Cheap-fit optimiser controls forwarded to :func:`fit_density_model`.
    **fit_kwargs
        Further keyword arguments forwarded to :func:`fit_density_model`.

    Returns
    -------
    list[DensityFitResult]
        One result per candidate, sorted by ``metrics['r1_like']`` ascending.
    """
    cands = list(candidates)
    if len(sites_per_candidate) > 0 and isinstance(sites_per_candidate[0], dict):
        per_candidate: list[list[dict[str, Any]]] = [
            list(sites_per_candidate)  # type: ignore[arg-type]
        ] * len(cands)
    else:
        per_candidate = [list(item) for item in sites_per_candidate]  # type: ignore[arg-type]
        if len(per_candidate) != len(cands):
            raise ValueError('sites_per_candidate must match the number of candidates')

    results: list[DensityFitResult] = []
    for spacegroup, site_specs in zip(cands, per_candidate):
        results.append(
            fit_density_model(
                observed_density,
                spacegroup,
                site_specs,
                symmetrize=symmetrize,
                supercell=supercell,
                cell_length=cell_length,
                maxiter=maxiter,
                popsize=popsize,
                seed=seed,
                **fit_kwargs,
            )
        )

    results.sort(key=lambda res: res.metrics['r1_like'])
    return results


def trajectory_to_symmetrized_density(
    trajectory: Trajectory,
    spacegroup: str | int | SpaceGroup,
    *,
    floating_specie: str | None = None,
    species: str | list[str] | None = None,
    resolution: float = 0.2,
    supercell: tuple[int, int, int] = (1, 1, 1),
    equilibration_cutoff: int = 0,
    drift_correction: bool = False,
    fixed_species: str | list[str] | None = None,
    origin_shift: np.ndarray | None = None,
) -> np.ndarray:
    """Build a fitting-ready symmetrised species density from a trajectory.

    Convenience wrapper: filter to the species of interest, drop
    pre-equilibration frames, bin to a grid, fold the supercell and symmetrise
    with ``spacegroup``.

    Notes
    -----
    gemdat already provides ``Trajectory.apply_drift_correction`` -- prefer it
    (via ``drift_correction=True``) over re-deriving an origin shift from an
    isotropic S-type fit. The explicit ``origin_shift`` argument is kept only to
    mirror the reference workflow, where the sulfur fit yields a small
    correction vector that is subtracted from the Li and O trajectory
    coordinates before their densities are built.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory.
    spacegroup : str | int | SpaceGroup
        Space group to symmetrise with.
    floating_specie : str | None, optional
        Diffusing species to build the density for (e.g. ``'Li'``). Alias:
        ``species`` (which also accepts a list).
    species : str | list[str] | None, optional
        Species selection, forwarded to ``Trajectory.filter``.
    resolution : float, optional
        Grid resolution in Angstrom for ``Trajectory.to_volume``.
    supercell : tuple[int, int, int], optional
        Supercell factor to fold out before symmetrising.
    equilibration_cutoff : int, optional
        Discard frames before this index.
    drift_correction : bool, optional
        Apply ``Trajectory.apply_drift_correction(fixed_species=fixed_species)``
        first.
    fixed_species : str | list[str] | None, optional
        Framework species for the drift correction.
    origin_shift : np.ndarray | None, optional
        Fractional vector subtracted from the (filtered) coordinates before
        binning -- mirrors the reference S-fit correction vector.

    Returns
    -------
    np.ndarray
        Symmetrised observed density, ready for :func:`fit_density_model` with
        ``symmetrize=False``.
    """
    selection = floating_specie if floating_specie is not None else species
    if selection is None:
        raise ValueError('provide either floating_specie or species')

    traj = trajectory
    if drift_correction:
        traj = traj.apply_drift_correction(fixed_species=fixed_species)

    sub = traj.filter(selection)
    if equilibration_cutoff:
        sub = sub[equilibration_cutoff:]

    if origin_shift is not None:
        shift = np.asarray(origin_shift, dtype=float)
        sub.coords = np.mod(np.array(sub.positions, dtype=float) - shift, 1.0)

    grid = np.asarray(sub.to_volume(resolution=resolution).data, dtype=float)

    sc = _as_shape(supercell)
    if sc != (1, 1, 1):
        grid = fold_supercell(grid, sc)

    return symmetrize_density(grid, spacegroup)
