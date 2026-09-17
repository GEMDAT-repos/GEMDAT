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
5. Compare candidate space groups on a parameter-penalised fit residual
   (:func:`rank_spacegroups`).

Grids follow the :class:`gemdat.volume.Volume` convention: voxel ``i`` of an axis
with ``n`` voxels is centred on fractional coordinate ``(i + 1/2) / n``, so the
data of ``Trajectory.to_volume`` can be used directly.

Deferred follow-ups (not in this module yet): the anisotropic-ADP O-style fit,
the per-species ordered S -> Li -> O workflow with origin-shift propagation, and
plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np
from pymatgen.symmetry.groups import SpaceGroup
from scipy.linalg import null_space
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
    'periodic_gaussian_density',
    'rank_spacegroups',
    'site_free_directions',
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


@lru_cache
def _op_arrays(number: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(rotations, translations)`` stacks, ``(K, 3, 3)`` and ``(K,
    3)``, for every symmetry operation of space group ``number``."""
    ops = SpaceGroup.from_int_number(number).symmetry_ops
    return (
        np.array([op.rotation_matrix for op in ops], dtype=float),
        np.array([op.translation_vector for op in ops], dtype=float),
    )


def _as_shape(grid_size: int | tuple[int, int, int]) -> Shape:
    """Normalise ``grid_size`` to a ``(nx, ny, nz)`` tuple."""
    if isinstance(grid_size, (int, np.integer)):
        return (int(grid_size), int(grid_size), int(grid_size))
    a, b, c = grid_size
    return (int(a), int(b), int(c))


def _voxel_axes(shape: Shape) -> list[np.ndarray]:
    """Fractional coordinates of the voxel centres along each axis; voxel ``i``
    of ``n`` sits at ``(i + 1/2) / n``, as in :class:`gemdat.volume.Volume`."""
    return [(np.arange(n, dtype=float) + 0.5) / n for n in shape]


def _fractional_grid(shape: Shape) -> np.ndarray:
    """Return the ``(N, 3)`` array of fractional coordinates of every voxel
    centre of a grid with the given ``shape``."""
    grids = np.meshgrid(*_voxel_axes(shape), indexing='ij')
    return np.stack([g.ravel() for g in grids], axis=-1)


def fold_supercell(grid: np.ndarray, supercell: int | tuple[int, int, int]) -> np.ndarray:
    """Fold a supercell density onto a single cell by summing the blocks.

    Generalises the reference ``fold_pdf`` (which only handled a 2x2x2 cubic
    supercell) to an arbitrary integer supercell.

    Parameters
    ----------
    grid : np.ndarray
        3D density on a grid whose dimensions are integer multiples of
        ``supercell``.
    supercell : int | tuple[int, int, int]
        Number of cells along each axis (a scalar means the same for all).

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
        ``i`` is centred on fractional coordinate ``(i + 1/2) / n``.
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

    rotations, translations = _op_arrays(sg.int_number)
    acc = np.zeros(grid.size, dtype=np.float64)
    for rot, trans in zip(rotations, translations):
        transformed = (coords @ rot.T + trans) % 1.0
        sample = (transformed * n - 0.5).T
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
    rotations, translations = _op_arrays(_as_spacegroup(spacegroup).int_number)
    return _orbit(rotations, translations, position, tol=tol)


def _orbit(
    rotations: np.ndarray,
    translations: np.ndarray,
    position: np.ndarray,
    *,
    tol: float = 1e-8,
) -> np.ndarray:
    """:func:`wyckoff_orbit` on raw ``(rotations, translations)`` stacks."""
    pos = np.asarray(position, dtype=float) % 1.0

    images = np.mod(np.einsum('kij,j->ki', rotations, pos) + translations, 1.0)

    # Deduplicate under periodic boundaries. Distinct orbit points are always
    # well separated, so rounding to the tolerance's decimal place (after
    # wrapping ~1.0 back to 0.0) is a safe, vectorised key.
    decimals = max(1, int(round(-np.log10(tol))))
    keyed = np.mod(np.round(images, decimals), 1.0)
    _, keep = np.unique(keyed, axis=0, return_index=True)
    return images[np.sort(keep)]


def site_free_directions(
    spacegroup: str | int | SpaceGroup,
    position: np.ndarray,
    *,
    tol: float = 1e-4,
) -> np.ndarray:
    """Displacement directions allowed by the site symmetry of a position.

    A site can move along ``v`` without changing its Wyckoff position when
    every operation of its site-symmetry group (the operations that map
    ``position`` onto itself modulo a lattice translation) leaves ``v``
    unchanged. These are the free coordinates of the Wyckoff position: ``x``
    for ``(x, x, x)``, ``x`` and ``z`` for ``(x, x, z)``, all three for the
    general position, none for a point such as Fm-3m 8c ``(1/4, 1/4, 1/4)``.

    Parameters
    ----------
    spacegroup : str | int | SpaceGroup
        Space group (symbol, number or object).
    position : np.ndarray
        Fractional coordinate, shape ``(3,)``.
    tol : float, optional
        Fractional distance below which an operation counts as mapping
        ``position`` onto itself.

    Returns
    -------
    np.ndarray
        Array of shape ``(k, 3)``, ``0 <= k <= 3``, one fractional direction per
        free coordinate, in row echelon form (e.g. ``[[1, 1, 1]]`` for
        ``(x, x, x)``) and scaled so the largest component of each is 1.
    """
    import sympy

    rotations, translations = _op_arrays(_as_spacegroup(spacegroup).int_number)
    pos = np.asarray(position, dtype=float) % 1.0

    delta = np.einsum('kij,j->ki', rotations, pos) + translations - pos
    delta -= np.round(delta)
    stabilizer = rotations[np.all(np.abs(delta) < tol, axis=1)]

    constraints = (stabilizer - np.eye(3)).reshape(-1, 3)
    basis = null_space(constraints).T
    if len(basis) == 0:
        return np.zeros((0, 3))
    rref, _ = sympy.Matrix(basis).rref(iszerofunc=lambda x: abs(x) < 1e-8)
    directions = np.array(rref, dtype=float)
    directions /= np.abs(directions).max(axis=1, keepdims=True)
    return np.round(directions, 12) + 0.0


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
    grid_x, grid_y, grid_z = np.meshgrid(*_voxel_axes(shape), indexing='ij')

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


def crystallographic_density_metrics(
    observed: np.ndarray,
    calculated: np.ndarray,
    *,
    n_params: int = 0,
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
    n_params : int, optional
        Number of fitted parameters, used for the goodness-of-fit ``gof`` and
        the information criteria ``aic`` / ``bic``.

    Returns
    -------
    dict[str, float]
        Keys: ``r1_like``, ``wr_like``, ``gof``, ``aic``, ``bic``, ``mse``,
        ``mae``, ``rmse``, ``pearson_r``, ``jensen_shannon``, ``n_voxels``.

    Notes
    -----
    ``aic`` and ``bic`` are the Gaussian-likelihood information criteria
    ``n ln(chi2 / n) + 2 k`` and ``n ln(chi2 / n) + k ln n``, with ``n`` the
    number of voxels, ``k = n_params`` and ``chi2`` the sum of
    squared residuals. Only differences between models fit to the *same*
    observed grid are meaningful. Neighbouring voxels are correlated, so ``n``
    overstates the independent data and the penalty is a lower bound.
    """
    obs = np.asarray(observed, dtype=float)
    calc = np.asarray(calculated, dtype=float)
    if obs.shape != calc.shape:
        raise ValueError('observed and calculated must have the same shape')

    obs = obs.ravel() / (np.sum(obs) + _EPS)
    calc = calc.ravel() / (np.sum(calc) + _EPS)
    residual = obs - calc

    r1_like = float(np.sum(np.abs(residual)) / (np.sum(np.abs(obs)) + _EPS))
    chi2 = float(np.sum(residual**2))
    wr_like = float(np.sqrt(chi2 / (np.sum(obs**2) + _EPS)))
    n_obs = obs.size
    dof = max(n_obs - int(n_params), 1)
    gof = float(np.sqrt(chi2 / dof))
    log_likelihood_term = n_obs * np.log(max(chi2 / n_obs, np.finfo(float).tiny))
    aic = float(log_likelihood_term + 2 * int(n_params))
    bic = float(log_likelihood_term + int(n_params) * np.log(n_obs))

    mse = float(np.mean(residual**2))
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(mse))
    pearson_r = float(np.corrcoef(obs, calc)[0, 1]) if n_obs > 1 else float('nan')

    p = obs + _EPS
    q = calc + _EPS
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    jensen_shannon = float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))

    return {
        'r1_like': r1_like,
        'wr_like': wr_like,
        'gof': gof,
        'aic': aic,
        'bic': bic,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'pearson_r': pearson_r,
        'jensen_shannon': jensen_shannon,
        'n_voxels': float(n_obs),
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
        Distance from the starting position in Angstrom -- the length scale at
        which the site is displaced from its higher-symmetry ideal position
        (issue #421 point 5). ``None`` for a fixed-position site or when no
        cell edge was supplied.
    """

    specie: str
    position: np.ndarray
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
    success : bool
        Optimiser success flag.
    params : np.ndarray
        Raw refined parameter vector.
    ranking_metrics : dict[str, float] | None
        Set by :func:`rank_spacegroups`: metrics of the model against the
        observed density *before* symmetrisation, which is the same grid for
        every candidate, so these (unlike ``metrics``) compare across groups.
    """

    spacegroup_symbol: str
    spacegroup_number: int
    sites: list[SiteFit]
    observed_density: np.ndarray
    model_density: np.ndarray
    metrics: dict[str, float]
    success: bool
    params: np.ndarray = field(repr=False)
    ranking_metrics: dict[str, float] | None = None


def _stick_breaking(raw: list[float]) -> np.ndarray:
    """Map ``n - 1`` numbers in ``(0, 1)`` to ``n`` non-negative fractions that
    sum to 1 (stick-breaking). For ``n == 2`` this reduces to
    ``[r, 1 - r]``, matching the reference's single ``alpha`` parameter."""
    fracs = np.asarray(raw, dtype=float)
    remaining = np.concatenate([[1.0], np.cumprod(1.0 - fracs)])
    return np.append(remaining[:-1] * fracs, remaining[-1])


def _free_directions(sg: SpaceGroup, position: np.ndarray, free_position: Any) -> np.ndarray:
    """Resolve a site spec's ``free_position`` to a ``(k, 3)`` array of
    displacement directions."""
    if free_position is None or isinstance(free_position, (bool, np.bool_)):
        if not free_position:
            return np.zeros((0, 3))
        directions = site_free_directions(sg, position)
        if len(directions) == 0:
            raise ValueError(
                f'position {np.round(position, 6).tolist()} has no free coordinates in '
                f'{sg.symbol}; to model a split site, pass the displacement '
                "direction(s) as free_position, e.g. 'free_position': (1, 1, 1)"
            )
        return directions

    directions = np.atleast_2d(np.asarray(free_position, dtype=float))
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError('free_position directions must have shape (3,) or (k, 3), k <= 3')
    if np.linalg.matrix_rank(directions) != len(directions):
        raise ValueError('free_position directions must be linearly independent')
    return directions / np.abs(directions).max(axis=1, keepdims=True)


class _MixtureModel:
    """Symmetry-constrained Gaussian-mixture model and its least-squares
    objective.

    Defined at module level (rather than as closures inside
    :func:`fit_density_model`) so that it can be pickled and sent to
    worker processes when ``workers != 1``.
    """

    def __init__(
        self,
        sg: SpaceGroup,
        specs: list[dict[str, Any]],
        observed: np.ndarray,
    ):
        # Raw op arrays rather than the SpaceGroup: scipy pickles the model per
        # task with workers != 1, and unpickling a SpaceGroup rebuilds its ops.
        self.rotations, self.translations = _op_arrays(sg.int_number)
        self.specs = specs
        self.shape: Shape = (observed.shape[0], observed.shape[1], observed.shape[2])
        self.observed = observed / (np.sum(observed) + _EPS)
        # Orbits of fixed-position sites never change, so compute them once.
        self.fixed_orbits = {
            idx: self.orbit(spec['position'])
            for idx, spec in enumerate(specs)
            if len(spec['directions']) == 0
        }

    @property
    def n_sites(self) -> int:
        return len(self.specs)

    def orbit(self, position: np.ndarray) -> np.ndarray:
        return _orbit(self.rotations, self.translations, position)

    def site_orbit(self, idx: int, position: np.ndarray) -> np.ndarray:
        orbit = self.fixed_orbits.get(idx)
        return self.orbit(position) if orbit is None else orbit

    def bounds(self) -> list[tuple[float, float]]:
        bounds: list[tuple[float, float]] = [spec['sigma_bounds'] for spec in self.specs]
        for spec in self.specs:
            md = spec['max_displacement']
            bounds.extend((-md, md) for _ in spec['directions'])
        bounds.extend((1e-3, 1.0 - 1e-3) for _ in range(self.n_sites - 1))
        return bounds

    def unpack(self, params: np.ndarray) -> tuple[list[float], list[np.ndarray], np.ndarray]:
        cursor = 0
        sigmas = [float(v) for v in params[cursor : cursor + self.n_sites]]
        cursor += self.n_sites
        positions: list[np.ndarray] = []
        for spec in self.specs:
            n_free = len(spec['directions'])
            shift = np.asarray(params[cursor : cursor + n_free], dtype=float)
            positions.append(spec['position'] + shift @ spec['directions'])
            cursor += n_free
        raw_fracs = [float(v) for v in params[cursor : cursor + (self.n_sites - 1)]]
        return sigmas, positions, _stick_breaking(raw_fracs)

    def density(self, params: np.ndarray) -> np.ndarray | None:
        sigmas, positions, alphas = self.unpack(params)
        total = np.zeros(self.shape, dtype=float)
        for idx, (sigma, pos, alpha) in enumerate(zip(sigmas, positions, alphas)):
            if sigma <= 0:
                return None
            single = periodic_gaussian_density(self.shape, self.site_orbit(idx, pos), sigma)
            norm = single.sum()
            if norm <= 0:
                return None
            total += alpha * (single / norm)
        return total

    def __call__(self, params: np.ndarray) -> float:
        grid = self.density(params)
        if grid is None:
            return np.inf
        return float(np.mean((grid - self.observed) ** 2))


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
    width ``sigma`` (and, optionally, one shift per free displacement
    direction of the representative), plus ``n_sites - 1`` inter-site
    occupancy fractions.

    Parameters
    ----------
    observed_density : np.ndarray
        3D density, e.g. ``Trajectory.filter('Li').to_volume(...).data`` or the
        output of :func:`trajectory_to_symmetrized_density`.
    spacegroup : str | int | SpaceGroup
        Candidate space group.
    sites : list[dict]
        One spec per crystallographic site, e.g.
        ``{'specie': 'Li', 'position': (0.25, 0.25, 0.25), 'free_position': False}``.
        Optional per-site keys: ``sigma_bounds``, ``max_displacement``.

        ``free_position`` selects which way the representative may move:
        ``True`` frees the coordinates allowed by its site symmetry (see
        :func:`site_free_directions`; e.g. only ``x`` for ``(x, x, x)``), so
        the site keeps its Wyckoff position, and raises if there are none.
        To model a site *splitting* off a special position, pass the
        displacement direction(s) instead, shape ``(3,)`` or ``(k, 3)``: e.g.
        ``(1, 1, 1)`` splits Fm-3m 8c ``(1/4, 1/4, 1/4)`` into 32f
        ``(x, x, x)``, and ``np.eye(3)`` frees all three coordinates. Each
        direction is scaled so its largest component is 1 and gets one
        parameter bounded by ``+/- max_displacement``.
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
        Global +/- bound (fractional units) for each free positional parameter.
    maxiter, popsize, tol, seed, polish, workers
        Passed to :func:`scipy.optimize.differential_evolution`. With
        ``workers != 1`` the population is evaluated in parallel
        (``updating='deferred'``).

    Returns
    -------
    DensityFitResult
    """
    sg = _as_spacegroup(spacegroup)
    obs = fold_supercell(observed_density, supercell or 1)
    if symmetrize:
        obs = symmetrize_density(obs, sg)

    specs: list[dict[str, Any]] = []
    for raw in sites:
        position = np.asarray(raw['position'], dtype=float) % 1.0
        specs.append(
            {
                'specie': str(raw['specie']),
                'position': position,
                'directions': _free_directions(sg, position, raw.get('free_position')),
                'sigma_bounds': tuple(raw.get('sigma_bounds', sigma_bounds)),
                'max_displacement': float(raw.get('max_displacement', max_displacement)),
            }
        )

    if len(specs) == 0:
        raise ValueError('sites must contain at least one site spec')

    model = _MixtureModel(sg, specs, obs)

    result = differential_evolution(
        model,
        model.bounds(),
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        seed=seed,
        polish=polish,
        workers=workers,
        updating='deferred' if workers != 1 else 'immediate',
    )

    sigmas, positions, alphas = model.unpack(result.x)
    model_density = model.density(result.x)
    assert model_density is not None
    metrics = crystallographic_density_metrics(obs, model_density, n_params=len(result.x))

    site_fits: list[SiteFit] = []
    for idx, (spec, sigma, pos, alpha) in enumerate(zip(specs, sigmas, positions, alphas)):
        u_iso: float | None = None
        displacement: float | None = None
        if cell_length is not None:
            u_iso = float((sigma * cell_length) ** 2)
            if len(spec['directions']):
                displacement = float(np.linalg.norm(pos - spec['position']) * cell_length)
        site_fits.append(
            SiteFit(
                specie=spec['specie'],
                position=np.asarray(pos, dtype=float) % 1.0,
                multiplicity=len(model.site_orbit(idx, pos)),
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
        success=bool(result.success),
        params=np.asarray(result.x, dtype=float),
    )


_RANKING_CRITERIA = ('bic', 'aic', 'gof', 'r1_like', 'wr_like')


def rank_spacegroups(
    observed_density: np.ndarray,
    candidates: list[str | int | SpaceGroup],
    sites_per_candidate: list[dict[str, Any]] | list[list[dict[str, Any]]],
    *,
    criterion: str = 'bic',
    supercell: tuple[int, int, int] | None = None,
    maxiter: int = 25,
    popsize: int = 10,
    seed: int | None = 0,
    **fit_kwargs: Any,
) -> list[DensityFitResult]:
    """Rank candidate space groups by how well a symmetry-constrained model
    fits.

    This is the density-fit analogue of "assume / rank the space group"
    (issue #421 point 1): for each candidate, symmetrise the observed density
    with that group and run a (deliberately cheap) :func:`fit_density_model`.

    Every candidate is then scored against the same grid, the observed density
    folded but *not* symmetrised (stored in ``ranking_metrics``), because each
    fit's own target is symmetrised with a different group. A lower-symmetry
    group with more parameters can always fit that grid at least as well, so
    the default criterion, ``bic``, penalises the parameter count: a group
    that only adds parameters (e.g. P-43m splitting I-43m 6b into two
    independent 3c + 3d orbits) ranks below the simpler group it reproduces.

    Parameters
    ----------
    observed_density : np.ndarray
        3D density.
    candidates : list
        Candidate space groups (symbols, numbers or objects).
    sites_per_candidate : list
        Either a single list of site specs (used for every candidate) or a list
        of such lists, one per candidate.
    criterion : str, optional
        Key of ``ranking_metrics`` to sort by, lowest first: ``'bic'``
        (default), ``'aic'`` (weaker penalty), ``'gof'`` (reduced chi,
        negligible penalty on large grids), or the unpenalised ``'r1_like'`` /
        ``'wr_like'``.
    supercell : tuple[int, int, int] | None, optional
        Forwarded to :func:`fit_density_model`, and folds the scoring grid.
    maxiter, popsize, seed
        Cheap-fit optimiser controls forwarded to :func:`fit_density_model`.
    **fit_kwargs
        Further keyword arguments forwarded to :func:`fit_density_model`
        (e.g. ``symmetrize``, ``cell_length``).

    Returns
    -------
    list[DensityFitResult]
        One result per candidate, sorted by ``ranking_metrics[criterion]``
        ascending.
    """
    if criterion not in _RANKING_CRITERIA:
        raise ValueError(f'criterion must be one of {_RANKING_CRITERIA}, got {criterion!r}')

    cands = list(candidates)
    if len(sites_per_candidate) > 0 and isinstance(sites_per_candidate[0], dict):
        per_candidate: list[list[dict[str, Any]]] = [
            list(sites_per_candidate)  # type: ignore[arg-type]
        ] * len(cands)
    else:
        per_candidate = [list(item) for item in sites_per_candidate]  # type: ignore[arg-type]
        if len(per_candidate) != len(cands):
            raise ValueError('sites_per_candidate must match the number of candidates')

    common = fold_supercell(observed_density, supercell or 1)

    results: list[DensityFitResult] = []
    for spacegroup, site_specs in zip(cands, per_candidate):
        result = fit_density_model(
            observed_density,
            spacegroup,
            site_specs,
            supercell=supercell,
            maxiter=maxiter,
            popsize=popsize,
            seed=seed,
            **fit_kwargs,
        )
        result.ranking_metrics = crystallographic_density_metrics(
            common, result.model_density, n_params=len(result.params)
        )
        results.append(result)

    results.sort(key=lambda res: res.ranking_metrics[criterion])  # type: ignore[index]
    return results


def trajectory_to_symmetrized_density(
    trajectory: Trajectory,
    spacegroup: str | int | SpaceGroup,
    *,
    floating_specie: str | list[str],
    resolution: float = 0.2,
    supercell: tuple[int, int, int] = (1, 1, 1),
    origin_shift: np.ndarray | None = None,
) -> np.ndarray:
    """Build a fitting-ready symmetrised species density from a trajectory.

    Convenience wrapper: filter to the species of interest, bin to a grid, fold
    the supercell and symmetrise with ``spacegroup``. Drop equilibration frames
    (``trajectory[n:]``) and apply ``Trajectory.apply_drift_correction`` first.

    Notes
    -----
    ``Trajectory.to_volume`` centres voxel ``i`` on ``(i + 1/2) / n``, the
    convention used throughout this module, so no half-voxel correction is
    needed. The fitted ``sigma`` still includes the histogram bin width:
    roughly ``sqrt(sigma_md**2 + h**2 / 12)`` for a voxel of fractional size
    ``h``.

    Prefer ``Trajectory.apply_drift_correction`` over re-deriving an origin
    shift from an isotropic S-type fit. The explicit ``origin_shift`` argument
    is kept only to mirror the reference workflow, where the sulfur fit yields a
    small correction vector that is subtracted from the Li and O trajectory
    coordinates before their densities are built.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory.
    spacegroup : str | int | SpaceGroup
        Space group to symmetrise with.
    floating_specie : str | list[str]
        Species to build the density for (e.g. ``'Li'``), forwarded to
        ``Trajectory.filter``.
    resolution : float, optional
        Grid resolution in Angstrom for ``Trajectory.to_volume``.
    supercell : tuple[int, int, int], optional
        Supercell factor to fold out before symmetrising.
    origin_shift : np.ndarray | None, optional
        Fractional vector subtracted from the (filtered) coordinates before
        binning -- mirrors the reference S-fit correction vector.

    Returns
    -------
    np.ndarray
        Symmetrised observed density, ready for :func:`fit_density_model` with
        ``symmetrize=False``.
    """
    sub = trajectory.filter(floating_specie)
    if origin_shift is not None:
        shift = np.asarray(origin_shift, dtype=float)
        sub.coords = np.mod(np.array(sub.positions, dtype=float) - shift, 1.0)

    grid = sub.to_volume(resolution=resolution).data
    return symmetrize_density(fold_supercell(grid, supercell), spacegroup)
