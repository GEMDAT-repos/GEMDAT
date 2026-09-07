"""This module contains the [Crystallizer][gemdat.crystallizer.Crystallizer],
which reconstructs a symmetry-fitted crystal structure (and cif file) from the
occupancy density of a molecular dynamics trajectory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .io import write_cif
from .utils import require_constant_lattice

if TYPE_CHECKING:
    from pathlib import Path

    from .trajectory import Trajectory


@dataclass
class SymmetryLevel:
    """One rung of the symmetry ladder: the space group found at a single
    symmetry tolerance.

    Parameters
    ----------
    symprec : float
        Symmetry tolerance (in Ångstrom) at which this level was evaluated.
    spacegroup_number : int | None
        International number of the space group, or `None` if the symmetry
        search failed at this tolerance.
    spacegroup_symbol : str | None
        International (Hermann-Mauguin) symbol, or `None` on failure.
    crystal_system : str | None
        Crystal system (e.g. `'cubic'`), or `None` on failure.
    n_symmetry_ops : int | None
        Number of symmetry operations in the space group, or `None` on failure.
    n_site_orbits : int | None
        Number of symmetry-distinct site groups (orbits), or `None` on failure.
    error : str | None
        The exception message if the symmetry search raised, else `None`.
    """

    symprec: float
    spacegroup_number: int | None = None
    spacegroup_symbol: str | None = None
    crystal_system: str | None = None
    n_symmetry_ops: int | None = None
    n_site_orbits: int | None = None
    error: str | None = None


def _build_symmetry_level(
    geometry: Structure,
    symprec: float,
    angle_tolerance: float,
) -> SymmetryLevel:
    """Evaluate the space group of `geometry` at a single `symprec`.

    Any exception raised by
    [pymatgen.symmetry.analyzer.SpacegroupAnalyzer][] is caught and
    stored on the returned level rather than propagated, so a sweep can
    continue past a tolerance that fails.
    """
    try:
        sga = SpacegroupAnalyzer(geometry, symprec=symprec, angle_tolerance=angle_tolerance)
        symmetrized = sga.get_symmetrized_structure()
        return SymmetryLevel(
            symprec=symprec,
            spacegroup_number=sga.get_space_group_number(),
            spacegroup_symbol=sga.get_space_group_symbol(),
            crystal_system=sga.get_crystal_system(),
            n_symmetry_ops=len(sga.get_symmetry_operations()),
            n_site_orbits=len(symmetrized.equivalent_indices),
        )
    except Exception as exc:  # noqa: BLE001 - record, don't drop, the failure
        return SymmetryLevel(symprec=symprec, error=str(exc))


def _normalize_spacegroup_symbol(symbol: str) -> str:
    """Strip all whitespace and lower-case a Hermann-Mauguin symbol for
    tolerant matching."""
    return ''.join(symbol.split()).lower()


def _match_target_spacegroup(
    levels: list[SymmetryLevel],
    target: int | str,
) -> SymmetryLevel | None:
    """Return the first level in `levels` matching `target` (an international
    number or a Hermann-Mauguin symbol), or `None` if nothing matches.

    `levels` is expected in ascending-symprec order, so the first match
    is the tightest tolerance that yields the target.
    """
    if isinstance(target, str):
        wanted = _normalize_spacegroup_symbol(target)
        for level in levels:
            if level.spacegroup_symbol is not None and (
                _normalize_spacegroup_symbol(level.spacegroup_symbol) == wanted
            ):
                return level
        return None

    for level in levels:
        if level.spacegroup_number == target:
            return level
    return None


def _format_ladder(levels: list[SymmetryLevel]) -> str:
    """Render a list of [SymmetryLevel][gemdat.crystallizer.SymmetryLevel] as a
    fixed-width text table."""
    headers = ('symprec (Å)', 'space group', '#', 'crystal system', '# orbits')
    rows: list[tuple[str, str, str, str, str]] = []
    for level in levels:
        if level.spacegroup_number is None:
            rows.append((f'{level.symprec:g}', 'failed', '-', '-', '-'))
        else:
            orbits = str(level.n_site_orbits) if level.n_site_orbits is not None else '-'
            rows.append(
                (
                    f'{level.symprec:g}',
                    level.spacegroup_symbol or '-',
                    str(level.spacegroup_number),
                    level.crystal_system or '-',
                    orbits,
                )
            )

    widths = [
        max([len(headers[i])] + [len(row[i]) for row in rows]) for i in range(len(headers))
    ]

    def _fmt(cells: tuple[str, ...]) -> str:
        return '  '.join(cell.ljust(width) for cell, width in zip(cells, widths))

    lines = [_fmt(headers), _fmt(tuple('-' * width for width in widths))]
    lines += [_fmt(row) for row in rows]
    return '\n'.join(lines)


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
    ladder : list[SymmetryLevel] | None
        The full symmetry ladder (one [SymmetryLevel][gemdat.crystallizer.Symmet
        ryLevel] per swept symprec), or `None` when `symprec` was set explicitly
        and no sweep was run.
    candidates : list[SymmetryLevel] | None
        The distinct space groups found across the sweep ("Top-X"), ranked by
        space-group number descending, each represented by the tightest symprec
        that produced it. `None` when no sweep was run.
    """

    structure: Structure
    spacegroup_symbol: str
    spacegroup_number: int
    symprec: float
    ladder: list[SymmetryLevel] | None = None
    candidates: list[SymmetryLevel] | None = None

    def format_ladder(self) -> str:
        """Return the symmetry ladder as a readable fixed-width table.

        Returns
        -------
        str
            One row per swept symprec, with columns `symprec (Å)`,
            `space group`, `#`, `crystal system` and `# orbits`. A tolerance at
            which the symmetry search failed shows `failed`/`-`.

        Raises
        ------
        ValueError
            If this result carries no ladder (i.e. `symprec` was set
            explicitly, so no sweep was performed).
        """
        if self.ladder is None:
            raise ValueError(
                'This result has no symmetry ladder because `symprec` was set '
                'explicitly. Call `Crystallizer.symmetry_ladder()` instead.'
            )
        return _format_ladder(self.ladder)


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
        """
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

    def symmetry_ladder(
        self,
        *,
        symprec_range: tuple[float, ...] = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5),
        angle_tolerance: float = 5.0,
        background_level: float = 0.1,
        **find_peaks_kwargs,
    ) -> list[SymmetryLevel]:
        """Fit the space group of the reconstructed geometry at a range of
        symmetry tolerances.

        This shows which sub-symmetries appear and at what precision (in
        Ångstrom): loosening `symprec` typically climbs from P1 towards the
        parent space group, and the number of symmetry-distinct site orbits
        drops accordingly.

        Parameters
        ----------
        symprec_range : tuple[float, ...]
            Symmetry tolerances (Ångstrom) to evaluate.
        angle_tolerance : float
            Angle tolerance (degrees) passed to
            [pymatgen.symmetry.analyzer.SpacegroupAnalyzer][].
        background_level : float
            Fraction of the maximum density used as the segmentation floor.
        **find_peaks_kwargs : dict
            Passed through to [gemdat.volume.Volume.find_peaks][].

        Returns
        -------
        list[SymmetryLevel]
            One [SymmetryLevel][gemdat.crystallizer.SymmetryLevel] per symprec,
            ordered by symprec ascending. A tolerance at which the symmetry
            search raised is kept with `None` fields and its `error` set.
        """
        geometry, _ = self._geometry_and_occupancies(
            background_level=background_level, **find_peaks_kwargs
        )
        return [
            _build_symmetry_level(geometry, symprec, angle_tolerance)
            for symprec in sorted(symprec_range)
        ]

    def crystallize(
        self,
        *,
        symprec: float | None = None,
        target_spacegroup: int | str | None = None,
        symprec_range: tuple[float, ...] = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5),
        angle_tolerance: float = 5.0,
        background_level: float = 0.1,
        **find_peaks_kwargs,
    ) -> CrystallizerResult:
        """Build the full structure and fit a space group.

        By default the symmetry tolerance is swept over `symprec_range` and the
        tolerance giving the highest space group number wins (ties broken
        towards the tightest tolerance). This can be overridden:

        - pass `symprec` to skip the sweep and fit at exactly that tolerance
          ("set it if you are sure what it should be");
        - pass `target_spacegroup` to pick the tightest tolerance in the sweep
          that yields that space group.

        Parameters
        ----------
        symprec : float | None
            If given, fit at exactly this tolerance (Ångstrom) and skip the
            sweep. `result.symprec` equals this value. Raises `ValueError` if
            the symmetry search fails at this tolerance (no silent fallback).
            Mutually exclusive with `target_spacegroup`.
        target_spacegroup : int | str | None
            If given, select the tightest (smallest) symprec in `symprec_range`
            whose fit matches this space group. An `int` is matched against the
            international number, a `str` against the Hermann-Mauguin symbol
            (case-insensitive, whitespace-insensitive). Raises `ValueError`
            listing what was found at each symprec if nothing matches. Mutually
            exclusive with `symprec`.
        symprec_range : tuple[float, ...]
            Symmetry tolerances (Ångstrom) to try when sweeping. The tolerance
            giving the highest space group number wins; ties are broken towards
            the tightest (smallest) tolerance.
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
            was run, `result.ladder` holds the full symmetry ladder and
            `result.candidates` the distinct space groups found, ranked by
            number descending.

        Raises
        ------
        ValueError
            If `symprec` and `target_spacegroup` are both given; if the fit
            fails at an explicit `symprec`; if no `target_spacegroup` match is
            found; or if no symprec in the sweep yields a symmetry.
        """
        if symprec is not None and target_spacegroup is not None:
            raise ValueError(
                'Pass either `symprec` or `target_spacegroup`, not both '
                '(they are contradictory).'
            )

        geometry, occupancies = self._geometry_and_occupancies(
            background_level=background_level, **find_peaks_kwargs
        )

        ladder: list[SymmetryLevel] | None = None
        candidates: list[SymmetryLevel] | None = None

        if symprec is not None:
            level = _build_symmetry_level(geometry, symprec, angle_tolerance)
            if level.spacegroup_number is None:
                raise ValueError(
                    f'Could not determine symmetry at symprec={symprec}: {level.error}'
                )
            best_symprec = symprec
        else:
            ladder = [
                _build_symmetry_level(geometry, sp, angle_tolerance)
                for sp in sorted(symprec_range)
            ]
            found = [lvl for lvl in ladder if lvl.spacegroup_number is not None]
            if not found:
                raise ValueError(
                    'Could not determine symmetry for any of the given symprec values.'
                )

            # Distinct space groups, tightest symprec each (ladder is
            # symprec-ascending), ranked by number descending -> "Top-X".
            seen: dict[int, SymmetryLevel] = {}
            for lvl in found:
                assert lvl.spacegroup_number is not None
                seen.setdefault(lvl.spacegroup_number, lvl)
            candidates = sorted(
                seen.values(),
                key=lambda lvl: lvl.spacegroup_number or 0,
                reverse=True,
            )

            if target_spacegroup is not None:
                match = _match_target_spacegroup(found, target_spacegroup)
                if match is None:
                    raise ValueError(
                        f'No symprec in the sweep produced space group '
                        f'{target_spacegroup!r}. Found:\n' + _format_ladder(ladder)
                    )
                best_symprec = match.symprec
            else:
                best = max(
                    found,
                    key=lambda lvl: (lvl.spacegroup_number or 0, -lvl.symprec),
                )
                best_symprec = best.symprec

        sga = SpacegroupAnalyzer(
            geometry, symprec=best_symprec, angle_tolerance=angle_tolerance
        )
        symmetrized = sga.get_symmetrized_structure()

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
            spacegroup_symbol=sga.get_space_group_symbol(),
            spacegroup_number=sga.get_space_group_number(),
            symprec=best_symprec,
            ladder=ladder,
            candidates=candidates,
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
