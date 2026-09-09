"""This module contains the [Crystallizer][gemdat.crystallizer.Crystallizer],
which reconstructs a symmetry-fitted crystal structure (and cif file) from the
occupancy density of a molecular dynamics trajectory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .caching import weak_lru_cache
from .io import write_cif
from .symmetry import SymmetryAnalyzer, SymmetryLevel, SymmetryRanking
from .utils import require_constant_lattice

if TYPE_CHECKING:
    from pathlib import Path

    from .trajectory import Trajectory


@dataclass
class CrystallizerResult:
    """Result of [crystallize][gemdat.crystallizer.Crystallizer.crystallize].

    Parameters
    ----------
    structure : Structure
        Full structure (static framework + density-derived mobile sites), with
        partial occupancies and occupancies averaged over symmetry-equivalent
        sites.
    spacegroup_symbol : str
        International symbol of the fitted space group.
    spacegroup_number : int
        International number of the fitted space group.
    symprec : float
        Symmetry tolerance (in Ångstrom) that produced the fit.
    ranking : SymmetryRanking | None
        The full symmetry ranking: one
        [SymmetryLevel][gemdat.symmetry.SymmetryLevel] per distinct space
        group found by the automatic scan, ordered by required `deviation`
        ascending, or one per tolerance (symprec ascending) when
        `symprec_range` was given explicitly. `None` when `symprec` was set
        explicitly and no sweep was run.
    """

    structure: Structure
    spacegroup_symbol: str
    spacegroup_number: int
    symprec: float
    ranking: SymmetryRanking | None = None

    def _require_ranking(self) -> SymmetryRanking:
        """Return the ranking, or explain why there is none."""
        if self.ranking is None:
            raise ValueError(
                'This result has no symmetry ranking because `symprec` was set '
                'explicitly. Call `Crystallizer.symmetry_ranking()` instead.'
            )
        return self.ranking

    @property
    def candidates(self) -> list[SymmetryLevel]:
        """The distinct space groups found across the sweep ("Top-X").

        Returns
        -------
        list[SymmetryLevel]
            One level per space group, ranked by space-group number
            descending, each represented by the tightest symprec that produced
            it.

        Raises
        ------
        ValueError
            If this result carries no ranking (i.e. `symprec` was set
            explicitly, so no sweep was performed). Test `result.ranking is
            None` to tell the two cases apart.
        """
        return self._require_ranking().candidates

    def format_ranking(self) -> str:
        """Return the symmetry ranking as a readable fixed-width table.

        See [SymmetryRanking.format][gemdat.symmetry.SymmetryRanking.format].

        Returns
        -------
        str

        Raises
        ------
        ValueError
            If this result carries no ranking (i.e. `symprec` was set
            explicitly, so no sweep was performed).
        """
        return self._require_ranking().format()


class Crystallizer:
    """Reconstruct a crystal structure from a trajectory's occupancy density.

    The pipeline is: the mobile species density is turned into candidate sites
    with partial occupancies, these are combined with the time-averaged static
    host framework, and the highest crystal symmetry consistent with the
    resulting structure is fitted. The result can be written to a cif file.
    """

    def __init__(
        self,
        *,
        trajectory: Trajectory,
        floating_specie: str,
        resolution: float = 0.2,
    ):
        """Set up the crystallizer.

        Parameters
        ----------
        trajectory : Trajectory
            Input trajectory
        floating_specie : str
            Symbol of the diffusing/mobile species, e.g. `'Li'`. All other
            species are treated as the static framework.
        resolution : float
            Minimum resolution for the density voxels in Ångstrom, passed to
            [gemdat.trajectory.Trajectory.to_volume][].
        """
        self.trajectory = trajectory
        self.floating_specie = floating_specie
        self.resolution = resolution

    @classmethod
    def from_trajectory(
        cls,
        trajectory: Trajectory,
        floating_specie: str,
        resolution: float = 0.2,
    ) -> Crystallizer:
        """Construct a [Crystallizer][gemdat.crystallizer.Crystallizer] from a
        trajectory.

        Parameters
        ----------
        trajectory : Trajectory
            Input trajectory
        floating_specie : str
            Symbol of the diffusing/mobile species, e.g. `'Li'`.
        resolution : float
            Minimum resolution for the density voxels in Ångstrom.

        Returns
        -------
        Crystallizer
        """
        return cls(
            trajectory=trajectory,
            floating_specie=floating_specie,
            resolution=resolution,
        )

    def _framework_species(self) -> list[str]:
        """Return the sorted list of non-floating species symbols."""
        symbols = {specie.symbol for specie in self.trajectory.species}  # type: ignore[union-attr]
        return sorted(symbols - {self.floating_specie})

    def mobile_sites(
        self,
        *,
        background_level: float = 0.1,
        **find_peaks_kwargs,
    ) -> Structure:
        """Extract the mobile-species sites from the density, with partial
        occupancies.

        Parameters
        ----------
        background_level : float
            Fraction of the maximum density used as the segmentation floor, see
            [gemdat.volume.Volume.to_structure][].
        **find_peaks_kwargs : dict
            Passed through to [gemdat.volume.Volume.find_peaks][].

        Returns
        -------
        structure : Structure
            Structure of mobile sites with partial occupancies.
        """
        mobile = self.trajectory.filter(self.floating_specie)
        volume = mobile.to_volume(resolution=self.resolution)
        return volume.to_structure(
            specie=self.floating_specie,
            background_level=background_level,
            return_occupancies=True,
            n_frames=len(mobile),
            **find_peaks_kwargs,
        )

    @require_constant_lattice
    def framework(self) -> Structure:
        """Return the static host framework as a fully-occupied structure.

        Fractional coordinates are averaged over all frames using a circular
        mean per axis, which is robust to atoms sitting on/near a periodic
        boundary.

        Returns
        -------
        structure : Structure
            Time-averaged framework structure (one site per framework atom).
        """
        framework_species = self._framework_species()
        lattice = self.trajectory.get_lattice()

        if not framework_species:
            return Structure(lattice=lattice, species=[], coords=np.empty((0, 3)))

        framework = self.trajectory.filter(framework_species)
        positions = framework.positions  # (n_frames, n_atoms, 3), fractional

        # Circular mean over frames, per atom and axis (PBC-safe).
        angles = np.angle(np.exp(2j * np.pi * positions).mean(axis=0))
        frac_coords = (angles / (2 * np.pi)) % 1

        species = [specie.symbol for specie in framework.species]  # type: ignore[union-attr]

        return Structure(lattice=lattice, species=species, coords=frac_coords)

    def _geometry_and_occupancies(
        self,
        *,
        background_level: float = 0.1,
        **find_peaks_kwargs,
    ) -> tuple[Structure, np.ndarray]:
        """Combine framework + mobile sites into a geometry-only structure.

        The returned structure has every site at full occupancy (element
        symbols only). Symmetry must be searched on this structure: the
        per-site occupancies are continuous floats, and feeding them to
        the symmetry finder would make every mobile site distinct and
        collapse the result to P1. The occupancies are returned
        separately, aligned to the structure's site order (framework
        sites are 1.0), so they can be averaged over the symmetry-
        equivalent classes afterwards.

        Extracting the density peaks dominates the cost of everything
        downstream, and the same geometry is reused by repeated
        `crystallize`/`symmetry_ranking`/`to_cif` calls, so the result
        is cached per set of arguments. Arguments that cannot be hashed
        (e.g. an explicit `peaks` array) simply bypass the cache.
        """
        key = tuple(sorted(find_peaks_kwargs.items()))
        try:
            hash(key)
        except TypeError:
            return self._compute_geometry_and_occupancies(
                background_level=background_level, **find_peaks_kwargs
            )
        return self._cached_geometry_and_occupancies(background_level, key)

    @weak_lru_cache()
    def _cached_geometry_and_occupancies(
        self,
        background_level: float,
        find_peaks_items: tuple[tuple[str, object], ...],
    ) -> tuple[Structure, np.ndarray]:
        """Cached wrapper around `_compute_geometry_and_occupancies`, taking
        the keyword arguments in a hashable form."""
        return self._compute_geometry_and_occupancies(
            background_level=background_level, **dict(find_peaks_items)
        )

    def _compute_geometry_and_occupancies(
        self,
        *,
        background_level: float = 0.1,
        **find_peaks_kwargs,
    ) -> tuple[Structure, np.ndarray]:
        """Do the work for `_geometry_and_occupancies`."""
        framework = self.framework()
        mobile = self.mobile_sites(background_level=background_level, **find_peaks_kwargs)

        symbols = [site.specie.symbol for site in framework]
        symbols += [next(iter(site.species.as_dict())) for site in mobile]

        occupancies = [1.0] * len(framework)
        occupancies += [site.species.num_atoms for site in mobile]

        # `framework` is empty when every species is the floating specie (e.g.
        # an already Li-filtered trajectory); its `frac_coords` then has no
        # column axis and can't be stacked, so drop empty parts.
        parts = [s.frac_coords for s in (framework, mobile) if len(s) > 0]
        coords = np.vstack(parts) if parts else np.empty((0, 3))

        geometry = Structure(
            lattice=framework.lattice,
            species=symbols,
            coords=coords,
        )
        return geometry, np.array(occupancies)

    def symmetry_ranking(
        self,
        *,
        symprec_range: tuple[float, ...] | None = None,
        symprec_min: float | None = None,
        symprec_max: float | None = None,
        n_samples: int | None = None,
        angle_tolerance: float = 5.0,
        background_level: float = 0.1,
        **find_peaks_kwargs,
    ) -> SymmetryRanking:
        """Rank the space groups the reconstructed geometry can adopt by the
        symmetry tolerance each one requires.

        This shows which sub-symmetries appear and at what precision (in
        Ångstrom): loosening `symprec` typically climbs from P1 towards the
        parent space group, and the number of symmetry-distinct site orbits
        drops accordingly.

        By default the tolerances are found automatically: the range is
        scanned for the space groups the geometry can adopt, and for each one
        the tolerance it actually requires is measured (`deviation`, in
        Ångstrom, and `angle_deviation`, in degrees) rather than read off the
        sampling grid. The result is one row per space group instead of one
        row per hand-picked tolerance. Pass `symprec_range` to evaluate a
        fixed list of tolerances instead.

        Parameters
        ----------
        symprec_range : tuple[float, ...] | None
            If given, evaluate exactly these tolerances (Ångstrom) and skip
            the automatic scan; the result then has one level per tolerance,
            including duplicated space groups. Mutually exclusive with the
            scan settings below.
        symprec_min : float | None
            Tightest tolerance (Ångstrom) of the automatic scan, default
            0.01 Å.
        symprec_max : float | None
            Loosest tolerance (Ångstrom) of the automatic scan, default 0.5 Å.
            Beyond ~0.5 Å the fit says more about the tolerance than about the
            structure.
        n_samples : int | None
            Number of log-spaced tolerances in the scan, default 40. A space
            group holding over a window narrower than the grid spacing can be
            missed; raise this to scan more finely.
        angle_tolerance : float
            Angle tolerance (degrees) passed to
            [pymatgen.symmetry.analyzer.SpacegroupAnalyzer][].
        background_level : float
            Fraction of the maximum density used as the segmentation floor.
        **find_peaks_kwargs : dict
            Passed through to [gemdat.volume.Volume.find_peaks][].

        Returns
        -------
        SymmetryRanking
            From the automatic scan: one
            [SymmetryLevel][gemdat.symmetry.SymmetryLevel] per distinct space
            group, ordered by `deviation` ascending, with `deviation`,
            `angle_deviation` and `n_observed` set. From an explicit
            `symprec_range`: one per tolerance, ordered by symprec ascending,
            where a tolerance at which the symmetry search raised is kept with
            `None` fields and its `error` set.

        Raises
        ------
        ValueError
            If `symprec_range` is combined with any of the scan settings.
        """
        geometry, _ = self._geometry_and_occupancies(
            background_level=background_level, **find_peaks_kwargs
        )
        analyzer = SymmetryAnalyzer(geometry, angle_tolerance=angle_tolerance)
        return analyzer.rank(
            symprec_range=symprec_range,
            symprec_min=symprec_min,
            symprec_max=symprec_max,
            n_samples=n_samples,
        )

    def crystallize(
        self,
        *,
        symprec: float | None = None,
        target_spacegroup: int | str | None = None,
        symprec_range: tuple[float, ...] | None = None,
        symprec_min: float | None = None,
        symprec_max: float | None = None,
        n_samples: int | None = None,
        angle_tolerance: float = 5.0,
        background_level: float = 0.1,
        **find_peaks_kwargs,
    ) -> CrystallizerResult:
        """Build the full structure and fit a space group.

        By default the symmetry tolerance is swept automatically (see
        [symmetry_ranking][gemdat.crystallizer.Crystallizer.symmetry_ranking]):
        every space group the geometry adopts between `symprec_min` and
        `symprec_max` is found, together with the deviation it requires, and
        the highest space group number wins (ties broken towards the tightest
        tolerance). This can be overridden:

        - pass `symprec` to skip the sweep and fit at exactly that tolerance
          ("set it if you are sure what it should be");
        - pass `target_spacegroup` to pick the tightest tolerance in the sweep
          that yields that space group;
        - pass `symprec_range` to sweep a fixed list of tolerances instead of
          scanning automatically.

        Parameters
        ----------
        symprec : float | None
            If given, fit at exactly this tolerance (Ångstrom) and skip the
            sweep. `result.symprec` equals this value. Raises `ValueError` if
            the symmetry search fails at this tolerance (no silent fallback).
            Mutually exclusive with `target_spacegroup` and with every setting
            that configures the sweep.
        target_spacegroup : int | str | None
            If given, select the tightest (smallest) symprec in the sweep whose
            fit matches this space group. An `int` is matched against the
            international number, a `str` against the Hermann-Mauguin symbol
            (case-insensitive, whitespace-insensitive). Raises `ValueError`
            listing what was found at each symprec if nothing matches. Mutually
            exclusive with `symprec`.
        symprec_range : tuple[float, ...] | None
            If given, sweep exactly these tolerances (Ångstrom) instead of
            scanning `symprec_min`..`symprec_max` automatically. The tolerance
            giving the highest space group number wins either way; ties are
            broken towards the tightest (smallest) tolerance. Mutually
            exclusive with the scan settings below.
        symprec_min : float | None
            Tightest tolerance (Ångstrom) of the automatic scan, default
            0.01 Å.
        symprec_max : float | None
            Loosest tolerance (Ångstrom) of the automatic scan, default 0.5 Å.
        n_samples : int | None
            Number of log-spaced tolerances in the automatic scan, default
            40.
        angle_tolerance : float
            Angle tolerance (degrees) passed to
            [pymatgen.symmetry.analyzer.SpacegroupAnalyzer][].
        background_level : float
            Fraction of the maximum density used as the segmentation floor.
        **find_peaks_kwargs : dict
            Passed through to [gemdat.volume.Volume.find_peaks][].

        Returns
        -------
        result : CrystallizerResult
            The symmetrized structure and the fitted space group. When a sweep
            was run, `result.ranking` holds the full symmetry ranking (ordered
            by required deviation ascending) and `result.candidates` the
            distinct space groups found, ranked by number descending.

        Raises
        ------
        ValueError
            If `symprec` is combined with any argument that configures the
            sweep; if the fit fails at an explicit `symprec`; if no
            `target_spacegroup` match is found; or if no symprec in the sweep
            yields a symmetry.
        """
        if symprec is not None:
            conflicting = [
                name
                for name, value in (
                    ('target_spacegroup', target_spacegroup),
                    ('symprec_range', symprec_range),
                    ('symprec_min', symprec_min),
                    ('symprec_max', symprec_max),
                    ('n_samples', n_samples),
                )
                if value is not None
            ]
            if conflicting:
                listed = ', '.join(f'`{name}`' for name in conflicting)
                raise ValueError(
                    f'`symprec` pins the tolerance, so it cannot be combined with '
                    f'{listed}; those select a tolerance from a sweep.'
                )

        geometry, occupancies = self._geometry_and_occupancies(
            background_level=background_level, **find_peaks_kwargs
        )

        analyzer = SymmetryAnalyzer(geometry, angle_tolerance=angle_tolerance)
        ranking: SymmetryRanking | None = None

        if symprec is not None:
            best_symprec = symprec
        else:
            ranking = analyzer.rank(
                symprec_range=symprec_range,
                symprec_min=symprec_min,
                symprec_max=symprec_max,
                n_samples=n_samples,
            )

            if target_spacegroup is not None:
                match = ranking.match(target_spacegroup)
                if match is None:
                    raise ValueError(
                        f'No symprec in the sweep produced space group '
                        f'{target_spacegroup!r}. Found:\n' + ranking.format()
                    )
                best_symprec = match.symprec
            else:
                # Raises if no tolerance in the sweep yielded a symmetry.
                best_symprec = ranking.best().symprec

        # The sweep already knows this tolerance works; an explicit `symprec`
        # is only checked here, so that it is fitted exactly once either way.
        try:
            sga = SpacegroupAnalyzer(
                geometry, symprec=best_symprec, angle_tolerance=angle_tolerance
            )
            symmetrized = sga.get_symmetrized_structure()
            spacegroup_symbol = sga.get_space_group_symbol()
            spacegroup_number = sga.get_space_group_number()
        except Exception as exc:
            raise ValueError(
                f'Could not determine symmetry at symprec={best_symprec}: {exc}'
            ) from exc

        # Average occupancy within each symmetry-equivalent class so that
        # equivalent sites are truly equivalent before the cif is written.
        # `equivalent_indices` indexes into `geometry`'s site order, which is
        # how `occupancies` is aligned.
        new_species: list[dict] = [{} for _ in range(len(symmetrized))]
        for group in symmetrized.equivalent_indices:
            symbol = symmetrized[group[0]].specie.symbol
            avg_occupancy = float(np.mean([occupancies[i] for i in group]))
            for i in group:
                new_species[i] = {symbol: avg_occupancy}

        structure = Structure(
            lattice=symmetrized.lattice,
            species=new_species,
            coords=[site.frac_coords for site in symmetrized],
        )

        return CrystallizerResult(
            structure=structure,
            spacegroup_symbol=spacegroup_symbol,
            spacegroup_number=spacegroup_number,
            symprec=best_symprec,
            ranking=ranking,
        )

    def to_cif(self, filename: Path | str, **kwargs) -> CrystallizerResult:
        """Crystallize and write the result to a cif file (with symmetry).

        Parameters
        ----------
        filename : Path | str
            Output filename (a `.cif` suffix is enforced).
        **kwargs : dict
            Passed through to [Crystallizer.crystallize][gemdat.crystallizer.Crys
            tallizer.crystallize].

        Returns
        -------
        result : CrystallizerResult
            The same result that was written to file.
        """
        result = self.crystallize(**kwargs)
        write_cif(result.structure, filename, symprec=result.symprec)
        return result
