"""This module contains the symmetry analysis used by the
[Crystallizer][gemdat.crystallizer.Crystallizer]: fitting the space group of a
structure as a function of the symmetry tolerance
([SymmetryAnalyzer][gemdat.symmetry.SymmetryAnalyzer]), and ranking the groups
that turn up by how much tolerance each one actually needs
([SymmetryRanking][gemdat.symmetry.SymmetryRanking])."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, overload

import numpy as np
from pymatgen.core import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from pymatgen.core import Structure


@dataclass
class SymmetryLevel:
    """One entry in a [SymmetryRanking][gemdat.symmetry.SymmetryRanking]: the
    space group found at a single symmetry tolerance.

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
    deviation : float | None
        How far the structure actually is from this symmetry: the largest
        distance (Ångstrom) an atom must move for the group's operations to
        hold exactly. This is a property of the structure, not of the search,
        so it does not depend on the `symprec` that found the group. `None`
        if it was not computed or could not be determined.
    angle_deviation : float | None
        The largest difference (degrees) between the input cell angles and
        those of the idealised cell for this group, i.e. the smallest
        `angle_tolerance` the fit needs. `None` if not computed.
    n_observed : int | None
        Only set by a scan (see
        [rank][gemdat.symmetry.SymmetryAnalyzer.rank]): how many of the scanned
        tolerances yielded this space group, as a measure of how robustly it
        holds. `None` when the tolerances were given explicitly.
    error : str | None
        The exception message if the symmetry search raised, or if the
        deviation could not be measured (in which case the space group was
        still determined), else `None`.
    """

    symprec: float
    spacegroup_number: int | None = None
    spacegroup_symbol: str | None = None
    crystal_system: str | None = None
    n_symmetry_ops: int | None = None
    n_site_orbits: int | None = None
    deviation: float | None = None
    angle_deviation: float | None = None
    n_observed: int | None = None
    error: str | None = None


def _normalize_spacegroup_symbol(symbol: str) -> str:
    """Strip all whitespace and lower-case a Hermann-Mauguin symbol for
    tolerant matching."""
    return ''.join(symbol.split()).lower()


def _rank_key(level: SymmetryLevel) -> tuple[bool, float, float]:
    """Sort key ranking levels by how little the structure has to give up:
    smallest positional deviation first, ties broken on the angular one.

    Levels whose deviation could not be measured sort last.
    """
    return (
        level.deviation is None,
        level.deviation if level.deviation is not None else 0.0,
        level.angle_deviation if level.angle_deviation is not None else 0.0,
    )


class SymmetryRanking(Sequence[SymmetryLevel]):
    """The space groups a structure was found to adopt, in the order they were
    evaluated.

    A read-only sequence of
    [SymmetryLevel][gemdat.symmetry.SymmetryLevel]s (iteration,
    indexing, `len`), plus the queries that make it a ranking: which
    groups turn up ([candidates][
    gemdat.symmetry.SymmetryRanking.candidates]), which one wins
    ([best][gemdat.symmetry.SymmetryRanking.best]), and whether a given
    group is reachable at all
    ([match][gemdat.symmetry.SymmetryRanking.match]).
    """

    def __init__(self, levels: Iterable[SymmetryLevel]):
        """Set up the ranking.

        Parameters
        ----------
        levels : Iterable[SymmetryLevel]
            Evaluated levels, in the order they should be reported.
        """
        self._levels = list(levels)

    def __len__(self) -> int:
        return len(self._levels)

    @overload
    def __getitem__(self, index: int) -> SymmetryLevel: ...

    @overload
    def __getitem__(self, index: slice) -> list[SymmetryLevel]: ...

    def __getitem__(self, index: int | slice) -> SymmetryLevel | list[SymmetryLevel]:
        return self._levels[index]

    def __repr__(self) -> str:
        groups = ', '.join(
            f'{level.spacegroup_symbol} ({level.spacegroup_number})' for level in self.found
        )
        return f'{type(self).__name__}({groups})'

    @property
    def found(self) -> list[SymmetryLevel]:
        """The levels at which a space group was actually determined.

        Returns
        -------
        list[SymmetryLevel]
            Levels whose symmetry search succeeded, in ranking order.
        """
        return [level for level in self if level.spacegroup_number is not None]

    @property
    def candidates(self) -> list[SymmetryLevel]:
        """The distinct space groups found ("Top-X").

        Returns
        -------
        list[SymmetryLevel]
            One level per space group, each the one that needed the tightest
            `symprec`, ranked by space-group number descending.
        """
        tightest: dict[int, SymmetryLevel] = {}
        for level in self.found:
            number = level.spacegroup_number
            assert number is not None  # `found` drops the failures
            current = tightest.get(number)
            if current is None or level.symprec < current.symprec:
                tightest[number] = level
        return [
            level
            for _, level in sorted(tightest.items(), key=lambda item: item[0], reverse=True)
        ]

    def match(self, target: int | str) -> SymmetryLevel | None:
        """Return the level matching `target`, at the tightest tolerance that
        reaches it.

        Parameters
        ----------
        target : int | str
            International number, or Hermann-Mauguin symbol (matched
            case-insensitively and ignoring whitespace).

        Returns
        -------
        SymmetryLevel | None
            The matching level, or `None` if the target was never reached.
        """
        if isinstance(target, str):
            wanted = _normalize_spacegroup_symbol(target)
            matches = [
                level
                for level in self.found
                if level.spacegroup_symbol is not None
                and _normalize_spacegroup_symbol(level.spacegroup_symbol) == wanted
            ]
        else:
            matches = [level for level in self.found if level.spacegroup_number == target]

        if not matches:
            return None
        return min(matches, key=lambda level: level.symprec)

    def best(self) -> SymmetryLevel:
        """Return the winning level: the highest space-group number, ties
        broken towards the tightest tolerance.

        Returns
        -------
        SymmetryLevel

        Raises
        ------
        ValueError
            If no level in the ranking yielded a space group.
        """
        if not self.found:
            raise ValueError(
                'Could not determine symmetry for any of the swept symprec values.'
            )
        return max(
            self.found,
            key=lambda level: (level.spacegroup_number or 0, -level.symprec),
        )

    def format(self) -> str:
        """Render the ranking as a readable fixed-width table.

        Returns
        -------
        str
            One row per level, with columns `symprec (Å)`, `deviation (Å)`,
            `angle dev (°)`, `space group`, `#`, `crystal system`, `# ops`,
            `# orbits` and `# hits`. A tolerance at which the symmetry search
            failed shows `failed`/`-`, as does any column that was not
            computed.
        """
        headers = (
            'symprec (Å)',
            'deviation (Å)',
            'angle dev (°)',
            'space group',
            '#',
            'crystal system',
            '# ops',
            '# orbits',
            '# hits',
        )

        def cell(value: object, spec: str = '') -> str:
            """Render one value, or `-` if it was never determined."""
            return '-' if value is None else format(value, spec)

        rows: list[tuple[str, ...]] = [
            (
                f'{level.symprec:g}',
                cell(level.deviation, '.4g'),
                cell(level.angle_deviation, '.3g'),
                cell(level.spacegroup_symbol) if level.spacegroup_number else 'failed',
                cell(level.spacegroup_number),
                cell(level.crystal_system),
                cell(level.n_symmetry_ops),
                cell(level.n_site_orbits),
                cell(level.n_observed),
            )
            for level in self
        ]

        widths = [
            max([len(headers[i])] + [len(row[i]) for row in rows]) for i in range(len(headers))
        ]

        def _fmt(cells: tuple[str, ...]) -> str:
            return '  '.join(cell.ljust(width) for cell, width in zip(cells, widths))

        lines = [_fmt(headers), _fmt(tuple('-' * width for width in widths))]
        lines += [_fmt(row) for row in rows]
        return '\n'.join(lines)


class SymmetryAnalyzer:
    """Fit the space group of a structure as a function of the symmetry
    tolerance.

    [pymatgen.symmetry.analyzer.SpacegroupAnalyzer][] can only be asked
    "which space group at this tolerance?" — never the inverse, and a
    single very loose tolerance is no shortcut, since it returns one
    group or fails outright.
    [rank][gemdat.symmetry.SymmetryAnalyzer.rank] therefore samples
    tolerances to find out *which* groups a structure can adopt, and
    then measures *how much* each one costs instead of searching for it.
    """

    def __init__(self, structure: Structure, *, angle_tolerance: float = 5.0):
        """Set up the analyzer.

        Parameters
        ----------
        structure : Structure
            Structure to fit. It must be geometry-only (all sites at full
            occupancy): partial occupancies make every site distinct to the
            symmetry finder and collapse the result to P1.
        angle_tolerance : float
            Angle tolerance (degrees) passed to
            [pymatgen.symmetry.analyzer.SpacegroupAnalyzer][].
        """
        self.structure = structure
        self.angle_tolerance = angle_tolerance

    def _deviation(self, dataset: Any) -> tuple[float | None, float | None, str | None]:
        """Measure how far the structure really is from the symmetry recorded
        in `dataset`.

        The symmetry finder only reports *whether* a group holds within the
        given tolerances, never how much slack it needed. That slack is
        recoverable from the symmetry dataset, which gives the operations in
        the input cell's own frame plus the idealised standard cell:

        - applying each operation and matching its images one-to-one onto the
          sites of the same species gives the largest displacement the group
          demands;
        - transforming the idealised standard lattice back into the input
          basis and comparing cell angles gives the angular slack.

        Both are properties of the structure rather than of the search, so
        they do not depend on the `symprec` at which the group was found.

        Parameters
        ----------
        dataset : SpglibDataset
            Symmetry dataset of the fit, from
            [pymatgen.symmetry.analyzer.SpacegroupAnalyzer.get_symmetry_dataset][].

        Returns
        -------
        tuple[float | None, float | None, str | None]
            Maximum displacement (Ångstrom), maximum angle difference
            (degrees), and `None`; or `(None, None, message)` if they could not
            be determined.
        """
        structure = self.structure
        try:
            frac_coords = structure.frac_coords
            symbols = np.array([site.specie.symbol for site in structure])
            # An operation may only map a site onto a site of the same species.
            forbidden = symbols[:, None] != symbols[None, :]

            # One operation at a time: the distances are measured with the
            # lattice metric under periodic boundaries, and batching
            # operations only makes the intermediates bigger without going
            # faster. Cost grows as operations x sites^2, which is why this is
            # not done for every tolerance in a scan.
            deviation = 0.0
            for rotation, translation in zip(dataset.rotations, dataset.translations):
                mapped = frac_coords @ np.asarray(rotation).T + np.asarray(translation)
                distances = structure.lattice.get_all_distances(mapped, frac_coords)
                distances[forbidden] = np.inf
                # An operation permutes the sites, so the images must be
                # matched one-to-one: taking each image's nearest site
                # independently could send two images to the same site and
                # understate the deviation.
                rows, columns = linear_sum_assignment(distances)
                deviation = max(deviation, float(distances[rows, columns].max(initial=0.0)))

            # (a_s b_s c_s) = (a b c) P^-1, up to the rigid rotation R that
            # spglib applies to the standard cell, so undo R and re-apply P.
            unrotated = dataset.std_lattice @ np.linalg.inv(dataset.std_rotation_matrix).T
            idealized = Lattice((unrotated.T @ dataset.transformation_matrix).T)
            angle_deviation = float(
                np.abs(np.array(structure.lattice.angles) - np.array(idealized.angles)).max()
            )
        except Exception as exc:  # noqa: BLE001 - a diagnostic, not the fit
            # Reported on the level's `error` rather than dropped, so that a
            # systematic failure does not just look like "not computed".
            return None, None, f'Could not measure the deviation: {exc}'

        return deviation, angle_deviation, None

    def _level(self, symprec: float, *, with_deviation: bool = False) -> SymmetryLevel:
        """Fit the space group at a single tolerance.

        Any exception raised by
        [pymatgen.symmetry.analyzer.SpacegroupAnalyzer][] is caught and stored
        on the returned level rather than propagated, so a sweep can continue
        past a tolerance that fails.

        Parameters
        ----------
        symprec : float
            Symmetry tolerance (Ångstrom).
        with_deviation : bool
            Also measure the tolerances the fit actually needs. This loops
            over the symmetry operations, so it is off by default.

        Returns
        -------
        SymmetryLevel
        """
        try:
            sga = SpacegroupAnalyzer(
                self.structure, symprec=symprec, angle_tolerance=self.angle_tolerance
            )
            dataset = sga.get_symmetry_dataset()
            if dataset is None:
                raise ValueError('spglib could not determine a symmetry dataset.')
            deviation, angle_deviation, error = (
                self._deviation(dataset) if with_deviation else (None, None, None)
            )
            return SymmetryLevel(
                symprec=symprec,
                spacegroup_number=sga.get_space_group_number(),
                spacegroup_symbol=sga.get_space_group_symbol(),
                crystal_system=sga.get_crystal_system(),
                # Both counts come out of the dataset that is already in hand;
                # `get_symmetry_operations()` and `get_symmetrized_structure()`
                # would rebuild the same information for every sampled
                # tolerance.
                n_symmetry_ops=len(dataset.rotations),
                n_site_orbits=len(set(dataset.equivalent_atoms)),
                deviation=deviation,
                angle_deviation=angle_deviation,
                error=error,
            )
        except Exception as exc:  # noqa: BLE001 - record, don't drop, the failure
            return SymmetryLevel(symprec=symprec, error=str(exc))

    def rank(
        self,
        *,
        symprec_range: tuple[float, ...] | None = None,
        symprec_min: float | None = None,
        symprec_max: float | None = None,
        n_samples: int | None = None,
    ) -> SymmetryRanking:
        """Rank the space groups the structure adopts, scanning the tolerance
        automatically unless a fixed list of tolerances is given.

        The scan enumerates the candidate groups by sampling a log-spaced grid
        of tolerances between `symprec_min` and `symprec_max`. What each group
        *costs* is then measured directly rather than searched for: `deviation`
        and `angle_deviation` record how far the structure actually sits from
        that symmetry. Those are properties of the structure, so unlike the
        `symprec` at which a group happens to turn up they are exact and
        independent of the sampling.

        The group is *not* a monotone function of `symprec`: a structure can
        flicker between two groups over a range of tolerances before settling
        on the higher-symmetry one. `n_observed` counts how many of the
        `n_samples` grid points gave each group, which distinguishes a group
        that holds over a wide range from one seen in a single narrow window.

        Passing `symprec_range` sweeps exactly those tolerances instead, one
        level each. The deviations are not measured then, since measuring is
        operations x sites^2 and a sweep repeats the same groups.

        Parameters
        ----------
        symprec_range : tuple[float, ...] | None
            If given, evaluate exactly these tolerances (Ångstrom) instead of
            scanning. Mutually exclusive with the scan settings below.
        symprec_min : float | None
            Tightest tolerance (Ångstrom) of the scan, default 0.01 Å.
        symprec_max : float | None
            Loosest tolerance (Ångstrom) of the scan, default 0.5 Å. Beyond
            ~0.5 Å the fit says more about the tolerance than about the
            structure.
        n_samples : int | None
            Number of log-spaced tolerances in the scan, default 40. A group
            occupying a window narrower than the grid spacing can be missed.

        Returns
        -------
        SymmetryRanking
            From the scan: one level per distinct space group found, ordered by
            `deviation` ascending (ties broken on `angle_deviation`), each
            carrying the tightest sampled `symprec` that produced it. Empty if
            no tolerance in the range yielded a symmetry. From an explicit
            `symprec_range`: one level per tolerance, ordered by symprec
            ascending.

        Raises
        ------
        ValueError
            If `symprec_range` is combined with any of the scan settings, or if
            the scan range or sample count is not usable.
        """
        if symprec_range is not None:
            conflicting = [
                name
                for name, value in (
                    ('symprec_min', symprec_min),
                    ('symprec_max', symprec_max),
                    ('n_samples', n_samples),
                )
                if value is not None
            ]
            if conflicting:
                listed = ', '.join(f'`{name}`' for name in conflicting)
                raise ValueError(
                    f'`symprec_range` lists the tolerances explicitly, so it cannot be '
                    f'combined with {listed}; those only configure the automatic scan.'
                )
            return SymmetryRanking([self._level(symprec) for symprec in sorted(symprec_range)])

        symprec_min = 0.01 if symprec_min is None else symprec_min
        symprec_max = 0.5 if symprec_max is None else symprec_max
        n_samples = 40 if n_samples is None else n_samples

        if symprec_min <= 0:
            raise ValueError('`symprec_min` must be positive.')
        if symprec_max < symprec_min:
            raise ValueError('`symprec_max` must not be smaller than `symprec_min`.')
        if n_samples < 2:
            raise ValueError('`n_samples` must be at least 2.')

        samples = [
            float(symprec) for symprec in np.geomspace(symprec_min, symprec_max, n_samples)
        ]
        levels = [self._level(symprec) for symprec in samples]

        counts: dict[int, int] = {}
        first_seen: dict[int, int] = {}
        for index, level in enumerate(levels):
            number = level.spacegroup_number
            if number is None:
                continue
            counts[number] = counts.get(number, 0) + 1
            first_seen.setdefault(number, index)

        # Re-fit each distinct group once at the tightest symprec that
        # produced it, this time measuring the tolerances it actually needs.
        thresholds = [
            replace(
                self._level(samples[index], with_deviation=True),
                n_observed=counts[number],
            )
            for number, index in first_seen.items()
        ]

        return SymmetryRanking(sorted(thresholds, key=_rank_key))
