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
    from collections.abc import Iterable
    from pathlib import Path

    from .trajectory import Trajectory


@dataclass
class CrystallizerResult:
    """Dataclass of a crystallized structure at a single symmetry tolerance.

    Returned by [Crystallizer.crystallize][gemdat.crystallizer.Crystallizer.crystallize],
    [Crystallizer.crystallize_at][gemdat.crystallizer.Crystallizer.crystallize_at]
    and
    [CrystallizerScan][gemdat.crystallizer.CrystallizerScan].
    Parameters
    ----------
    structure : Structure
        Full structure (static framework + density-derived mobile sites), with
        occupancies averaged over symmetry-equivalent sites. If ``use_density``
        is True the mobile sites carry their density-derived partial
        occupancies (the MD time-fraction the site is occupied); if False every
        mobile site is emitted at full occupancy 1.0 and the structure is the
        idealised site framework without MD occupancy statistics. Framework
        sites are always fully occupied.
    spacegroup_symbol : str
        International symbol of the fitted space group.
    spacegroup_number : int
        International number of the fitted space group.
    symprec : float
        Symmetry tolerance (in Ångstrom) that produced the fit.
    use_density : bool
        Whether density-derived partial occupancies were written into
        ``structure`` (True) or every mobile site was set to full occupancy
        (False).
    """

    structure: Structure
    spacegroup_symbol: str
    spacegroup_number: int
    symprec: float
    use_density: bool = True

    def to_cif(self, filename: Path | str) -> None:
        """Write this structure to a cif file, with its symmetry.

        Parameters
        ----------
        filename : Path | str
            Output filename (a `.cif` suffix is enforced).
        """
        write_cif(self.structure, filename, symprec=self.symprec)


def _fit(
    geometry: Structure,
    occupancies: np.ndarray,
    *,
    symprec: float,
    angle_tolerance: float,
    use_density: bool = True,
) -> CrystallizerResult:
    """Fit `geometry` at one tolerance and assemble the result.

    Parameters
    ----------
    geometry : Structure
        Geometry-only structure (all sites at full occupancy).
    occupancies : np.ndarray
        Occupancy per site, aligned to `geometry`'s site order.
    symprec : float
        Symmetry tolerance (Ångstrom).
    angle_tolerance : float
        Angle tolerance (degrees).
    use_density : bool
        Whether `occupancies` carries the density-derived occupancies, recorded
        on the result. It does not change the fit -- the flag is applied when
        the occupancies are built, see
        [Crystallizer.crystallize][gemdat.crystallizer.Crystallizer.crystallize].

    Returns
    -------
    CrystallizerResult
    """
    try:
        sga = SpacegroupAnalyzer(geometry, symprec=symprec, angle_tolerance=angle_tolerance)
        symmetrized = sga.get_symmetrized_structure()
        spacegroup_symbol = sga.get_space_group_symbol()
        spacegroup_number = sga.get_space_group_number()
    except Exception as exc:
        raise ValueError(f'Could not determine symmetry at symprec={symprec}: {exc}') from exc

    # Average occupancy within each symmetry-equivalent class so that
    # equivalent sites are truly equivalent before the cif is written.
    # `equivalent_indices` indexes into `geometry`'s site order, which is how
    # `occupancies` is aligned.
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
        symprec=symprec,
        use_density=use_density,
    )


class CrystallizerScan:
    """A reconstructed geometry together with the space groups it can adopt.

    Returned by [Crystallizer.scan][gemdat.crystallizer.Crystallizer.scan] and
    [Crystallizer.scan_at][gemdat.crystallizer.Crystallizer.scan_at]. The
    expensive parts — extracting the density peaks and sweeping the symmetry
    tolerance — happen once, when the scan is made; picking a space group off
    it afterwards is a single symmetry fit:

    ```python
    scan = crystallizer.scan()
    print(scan.format())

    scan.best().to_cif('crystallized.cif')
    scan.at_spacegroup('Cm').to_cif('monoclinic.cif')
    for level in scan.candidates:
        scan.at_level(level).to_cif(f'{level.spacegroup_number}.cif')
    ```
    """

    def __init__(
        self,
        *,
        geometry: Structure,
        occupancies: np.ndarray,
        ranking: SymmetryRanking,
        angle_tolerance: float = 5.0,
        use_density: bool = True,
    ):
        """Set up the scan.

        Built by [Crystallizer][gemdat.crystallizer.Crystallizer], not directly.

        Parameters
        ----------
        geometry : Structure
            Geometry-only structure that was fitted (framework + mobile sites,
            all at full occupancy).
        occupancies : np.ndarray
            Occupancy per site, aligned to `geometry`'s site order.
        ranking : SymmetryRanking
            The space groups found for `geometry`.
        angle_tolerance : float
            Angle tolerance (degrees) the ranking was produced with, reused by
            every fit taken off this scan.
        use_density : bool
            Whether `occupancies` carries the density-derived occupancies,
            recorded on every result taken off this scan.
        """
        self.geometry = geometry
        self.occupancies = occupancies
        self.ranking = ranking
        self.angle_tolerance = angle_tolerance
        self.use_density = use_density

    def __repr__(self) -> str:
        return f'{type(self).__name__}({len(self.geometry)} sites, {self.ranking!r})'

    @property
    def candidates(self) -> list[SymmetryLevel]:
        """The distinct space groups found ("Top-X").

        Returns
        -------
        list[SymmetryLevel]
            One level per space group, ranked by space-group number
            descending, each represented by the tightest symprec that produced
            it. See
            [SymmetryRanking.candidates][gemdat.symmetry.SymmetryRanking.candidates].
        """
        return self.ranking.candidates

    def format(self) -> str:
        """Return the symmetry ranking as a readable fixed-width table.

        See [SymmetryRanking.format][gemdat.symmetry.SymmetryRanking.format].

        Returns
        -------
        str
        """
        return self.ranking.format()

    def best(self) -> CrystallizerResult:
        """Crystallize the highest space group found.

        Ties are broken towards the tightest tolerance.

        Returns
        -------
        CrystallizerResult

        Raises
        ------
        ValueError
            If no tolerance in the scan yielded a symmetry.
        """
        return self.at(self.ranking.best().symprec)

    def at(self, symprec: float) -> CrystallizerResult:
        """Crystallize at exactly this tolerance.

        The tolerance need not be one the scan visited.

        Parameters
        ----------
        symprec : float
            Symmetry tolerance (Ångstrom) to fit at.

        Returns
        -------
        CrystallizerResult

        Raises
        ------
        ValueError
            If the symmetry search fails at this tolerance.
        """
        return _fit(
            self.geometry,
            self.occupancies,
            symprec=symprec,
            angle_tolerance=self.angle_tolerance,
            use_density=self.use_density,
        )

    def at_level(self, level: SymmetryLevel) -> CrystallizerResult:
        """Crystallize the space group of one row of this scan's ranking.

        The level records the tolerance its space group was found at, so a
        group picked off the ranking table is crystallized without translating
        it back into a tolerance.

        Parameters
        ----------
        level : SymmetryLevel
            A row of `self.ranking`, e.g. from
            [candidates][gemdat.crystallizer.CrystallizerScan.candidates].

        Returns
        -------
        CrystallizerResult

        Raises
        ------
        ValueError
            If the level is not one of this scan's, or if it found no space
            group.
        """
        if level not in self.ranking:
            raise ValueError(
                'The given level is not part of this scan. A level fixes a '
                'tolerance for the geometry it was ranked on, so it can only be '
                'crystallized by the scan that produced it.'
            )
        if level.spacegroup_number is None:
            raise ValueError(
                f'The given level found no space group at '
                f'symprec={level.symprec:g} ({level.error}), so there is '
                f'nothing to crystallize at its tolerance.'
            )
        return self.at(level.symprec)

    def at_spacegroup(self, spacegroup: int | str) -> CrystallizerResult:
        """Crystallize the space group you name.

        The tightest tolerance in the scan that reaches it is used.

        Parameters
        ----------
        spacegroup : int | str
            An `int` is matched against the international number, a `str`
            against the Hermann-Mauguin symbol (case-insensitive,
            whitespace-insensitive).

        Returns
        -------
        CrystallizerResult

        Raises
        ------
        ValueError
            If no tolerance in the scan produces this space group; the message
            lists what was found instead.
        """
        match = self.ranking.match(spacegroup)
        if match is None:
            raise ValueError(
                f'No symprec in this scan produced space group '
                f'{spacegroup!r}. Found:\n' + self.format()
            )
        return self.at(match.symprec)


class Crystallizer:
    """Reconstruct a crystal structure from a trajectory's occupancy density.

    The pipeline is: the mobile species density is turned into candidate sites
    with partial occupancies, these are combined with the time-averaged static
    host framework, and the crystal symmetry of the resulting structure is
    fitted. The result can be written to a cif file.

    Which symmetry you get depends on the tolerance it is fitted at, so each
    way of choosing that tolerance is its own entry point:

    - [crystallize][gemdat.crystallizer.Crystallizer.crystallize] (and
      [to_cif][gemdat.crystallizer.Crystallizer.to_cif]) scan the tolerance and
      take the highest space group found — the default answer;
    - [scan][gemdat.crystallizer.Crystallizer.scan] /
      [scan_at][gemdat.crystallizer.Crystallizer.scan_at] hand back the whole
      [CrystallizerScan][gemdat.crystallizer.CrystallizerScan], which reports
      every space group the geometry reaches and crystallizes any of them;
    - [crystallize_at][gemdat.crystallizer.Crystallizer.crystallize_at] fits a
      tolerance you already know, without scanning.

    Incorporating the density-derived occupancies (``use_density=True``, the
    default) gives realistic partial site occupancies, but those numbers carry
    the statistical noise of the finite MD run. Switching it off
    (``use_density=False``) keeps only the geometry: sites are still *located*
    from the density peaks (the only way to find them), but every mobile site
    is emitted at full occupancy 1.0, giving a clean idealised site set without
    MD occupancy statistics. The fitted symmetry is identical either way -- the
    symmetry search always runs on the geometry alone.
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
        with_occupancies: bool = True,
        **find_peaks_kwargs,
    ) -> Structure:
        """Extract the mobile-species sites from the density.

        Parameters
        ----------
        background_level : float
            Fraction of the maximum density used as the segmentation floor, see
            [gemdat.volume.Volume.to_structure][].
        with_occupancies : bool
            If True (default), each site carries its density-derived partial
            occupancy (the MD time-fraction the site is occupied). If False,
            every site is returned at nominal full occupancy -- only the site
            positions (density-peak centroids) are kept.
        **find_peaks_kwargs : dict
            Passed through to [gemdat.volume.Volume.find_peaks][].

        Returns
        -------
        structure : Structure
            Structure of mobile sites. Occupancies are partial floats when
            ``with_occupancies`` is True and 1.0 otherwise.
        """
        mobile = self.trajectory.filter(self.floating_specie)
        volume = mobile.to_volume(resolution=self.resolution)
        return volume.to_structure(
            specie=self.floating_specie,
            background_level=background_level,
            return_occupancies=with_occupancies,
            n_frames=len(mobile) if with_occupancies else None,
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
        use_density: bool = True,
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
        `crystallize`/`scan`/`to_cif` calls, so the result is cached per
        set of arguments. Arguments that cannot be hashed (e.g. an
        explicit `peaks` array) simply bypass the cache.

        If ``use_density`` is False the mobile occupancies are all set to
        1.0. The sites are still located exactly the same way (from the
        density peaks), only the occupancy weighting is dropped, so the
        geometry structure -- and therefore the fitted symmetry -- does not
        depend on this flag. It is applied after the cache, so toggling it
        never re-extracts the peaks.
        """
        key = tuple(sorted(find_peaks_kwargs.items()))
        try:
            hash(key)
        except TypeError:
            geometry, occupancies = self._compute_geometry_and_occupancies(
                background_level=background_level, **find_peaks_kwargs
            )
        else:
            geometry, occupancies = self._cached_geometry_and_occupancies(background_level, key)

        if not use_density:
            occupancies = np.ones_like(occupancies)

        return geometry, occupancies

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

    def scan(
        self,
        *,
        symprec_min: float | None = None,
        symprec_max: float | None = None,
        n_samples: int | None = None,
        angle_tolerance: float = 5.0,
        background_level: float = 0.1,
        use_density: bool = True,
        **find_peaks_kwargs,
    ) -> CrystallizerScan:
        """Reconstruct the geometry and rank the space groups it can adopt.

        This shows which sub-symmetries the data supports and at what precision
        (in Ångstrom): loosening `symprec` typically climbs from P1 towards the
        parent space group, and the number of symmetry-distinct site orbits
        drops accordingly.

        The tolerances are found automatically: the range is scanned for the
        space groups the geometry can adopt, and for each one the tolerance it
        actually requires is measured (`deviation`, in Ångstrom, and
        `angle_deviation`, in degrees) rather than read off the sampling grid.
        The ranking therefore has one row per space group. Pass a fixed list of
        tolerances to [scan_at][gemdat.crystallizer.Crystallizer.scan_at]
        instead to get one row per tolerance.

        The returned scan crystallizes any of the space groups it found, so
        this is the way to write out more than one fit:

        ```python
        scan = crystallizer.scan()
        print(scan.format())
        scan.best().to_cif('crystallized.cif')
        scan.at_spacegroup(217).to_cif('cubic.cif')
        ```

        Parameters
        ----------
        symprec_min : float | None
            Tightest tolerance (Ångstrom) of the scan, default 0.01 Å.
        symprec_max : float | None
            Loosest tolerance (Ångstrom) of the scan, default 0.5 Å. Beyond
            ~0.5 Å the fit says more about the tolerance than about the
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
        use_density : bool
            Whether to incorporate the density-derived occupancy information.
            If True (default), the mobile sites carry their partial
            occupancies, averaged per symmetry orbit -- realistic occupancies,
            but they carry the MD occupancy noise. If False, the mobile sites
            are still located from the density peaks but emitted at full
            occupancy 1.0 -- a clean idealised site set without MD occupancy
            statistics. The fitted space group is unaffected by this flag.
        **find_peaks_kwargs : dict
            Passed through to [gemdat.volume.Volume.find_peaks][].

        Returns
        -------
        CrystallizerScan
            The reconstructed geometry and its ranking: one
            [SymmetryLevel][gemdat.symmetry.SymmetryLevel] per distinct space
            group, ordered by `deviation` ascending, with `deviation`,
            `angle_deviation` and `n_observed` set.
        """
        geometry, occupancies, analyzer = self._geometry_and_analyzer(
            angle_tolerance=angle_tolerance,
            background_level=background_level,
            use_density=use_density,
            **find_peaks_kwargs,
        )
        ranking = analyzer.rank(
            symprec_min=symprec_min,
            symprec_max=symprec_max,
            n_samples=n_samples,
        )
        return CrystallizerScan(
            geometry=geometry,
            occupancies=occupancies,
            ranking=ranking,
            angle_tolerance=angle_tolerance,
            use_density=use_density,
        )

    def scan_at(
        self,
        symprecs: Iterable[float],
        *,
        angle_tolerance: float = 5.0,
        background_level: float = 0.1,
        use_density: bool = True,
        **find_peaks_kwargs,
    ) -> CrystallizerScan:
        """Reconstruct the geometry and fit it at each of the given tolerances.

        Where [scan][gemdat.crystallizer.Crystallizer.scan] finds the
        tolerances itself and reports one row per space group, this reports one
        row per tolerance handed to it, including repeated space groups. The
        deviations are not measured.

        Parameters
        ----------
        symprecs : Iterable[float]
            Tolerances (Ångstrom) to evaluate.
        angle_tolerance : float
            Angle tolerance (degrees) passed to
            [pymatgen.symmetry.analyzer.SpacegroupAnalyzer][].
        background_level : float
            Fraction of the maximum density used as the segmentation floor.
        use_density : bool
            Whether to incorporate the density-derived occupancy information.
            If True (default), the mobile sites carry their partial
            occupancies, averaged per symmetry orbit -- realistic occupancies,
            but they carry the MD occupancy noise. If False, the mobile sites
            are still located from the density peaks but emitted at full
            occupancy 1.0 -- a clean idealised site set without MD occupancy
            statistics. The fitted space group is unaffected by this flag.
        **find_peaks_kwargs : dict
            Passed through to [gemdat.volume.Volume.find_peaks][].

        Returns
        -------
        CrystallizerScan
            The reconstructed geometry and its ranking: one
            [SymmetryLevel][gemdat.symmetry.SymmetryLevel] per tolerance,
            ordered by symprec ascending, where a tolerance at which the
            symmetry search raised is kept with `None` fields and its `error`
            set.
        """
        geometry, occupancies, analyzer = self._geometry_and_analyzer(
            angle_tolerance=angle_tolerance,
            background_level=background_level,
            use_density=use_density,
            **find_peaks_kwargs,
        )
        return CrystallizerScan(
            geometry=geometry,
            occupancies=occupancies,
            ranking=analyzer.rank_at(symprecs),
            angle_tolerance=angle_tolerance,
            use_density=use_density,
        )

    def crystallize(self, **kwargs) -> CrystallizerResult:
        """Build the full structure and fit the highest space group it
        supports.

        This is [scan][gemdat.crystallizer.Crystallizer.scan] followed by
        [CrystallizerScan.best][gemdat.crystallizer.CrystallizerScan.best]: the
        symmetry tolerance is scanned, and the highest space group number found
        wins (ties broken towards the tightest tolerance). Keep the
        [scan][gemdat.crystallizer.Crystallizer.scan] itself to see the
        alternatives, or to crystallize one of them;
        [crystallize_at][gemdat.crystallizer.Crystallizer.crystallize_at] fits
        a tolerance you already know without scanning at all.

        Parameters
        ----------
        **kwargs : dict
            Passed through to [scan][gemdat.crystallizer.Crystallizer.scan],
            e.g. `symprec_min=`, `symprec_max=`, `n_samples=`,
            `angle_tolerance=`, `background_level=`, `use_density=` and the
            peak-finding arguments.

        Returns
        -------
        result : CrystallizerResult
            The symmetrized structure and the fitted space group.

        Raises
        ------
        ValueError
            If no symprec in the scan yields a symmetry.
        """
        return self.scan(**kwargs).best()

    def crystallize_at(
        self,
        symprec: float,
        *,
        angle_tolerance: float = 5.0,
        background_level: float = 0.1,
        use_density: bool = True,
        **find_peaks_kwargs,
    ) -> CrystallizerResult:
        """Build the full structure and fit it at exactly this tolerance.

        Nothing is scanned, so this costs a single symmetry fit — use it when
        you already know the tolerance you want.

        Parameters
        ----------
        symprec : float
            Symmetry tolerance (Ångstrom) to fit at.
        angle_tolerance : float
            Angle tolerance (degrees) passed to
            [pymatgen.symmetry.analyzer.SpacegroupAnalyzer][].
        background_level : float
            Fraction of the maximum density used as the segmentation floor.
        use_density : bool
            Whether to incorporate the density-derived occupancy information.
            If True (default), the mobile sites carry their partial
            occupancies, averaged per symmetry orbit -- realistic occupancies,
            but they carry the MD occupancy noise. If False, the mobile sites
            are still located from the density peaks but emitted at full
            occupancy 1.0 -- a clean idealised site set without MD occupancy
            statistics. The fitted space group is unaffected by this flag.
        **find_peaks_kwargs : dict
            Passed through to [gemdat.volume.Volume.find_peaks][].

        Returns
        -------
        result : CrystallizerResult
            The symmetrized structure and the fitted space group.

        Raises
        ------
        ValueError
            If the symmetry search fails at this tolerance.
        """
        geometry, occupancies = self._geometry_and_occupancies(
            background_level=background_level,
            use_density=use_density,
            **find_peaks_kwargs,
        )
        return _fit(
            geometry,
            occupancies,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
            use_density=use_density,
        )

    def _geometry_and_analyzer(
        self,
        *,
        angle_tolerance: float,
        background_level: float,
        use_density: bool = True,
        **find_peaks_kwargs,
    ) -> tuple[Structure, np.ndarray, SymmetryAnalyzer]:
        """Reconstruct the geometry and wrap it in a
        [SymmetryAnalyzer][gemdat.symmetry.SymmetryAnalyzer]."""
        geometry, occupancies = self._geometry_and_occupancies(
            background_level=background_level,
            use_density=use_density,
            **find_peaks_kwargs,
        )
        analyzer = SymmetryAnalyzer(geometry, angle_tolerance=angle_tolerance)
        return geometry, occupancies, analyzer

    def to_cif(self, filename: Path | str, **kwargs) -> CrystallizerResult:
        """Crystallize and write the result to a cif file (with symmetry).

        This is [crystallize][gemdat.crystallizer.Crystallizer.crystallize]
        followed by
        [CrystallizerResult.to_cif][gemdat.crystallizer.CrystallizerResult.to_cif].
        To write a fit chosen some other way, take it off a
        [scan][gemdat.crystallizer.Crystallizer.scan] and let it write itself,
        e.g. one file per space group found:

        ```python
        scan = crystallizer.scan()
        for level in scan.candidates:
            scan.at_level(level).to_cif(f'{level.spacegroup_number}.cif')
        ```

        Parameters
        ----------
        filename : Path | str
            Output filename (a `.cif` suffix is enforced).
        **kwargs : dict
            Passed through to
            [crystallize][gemdat.crystallizer.Crystallizer.crystallize].

        Returns
        -------
        result : CrystallizerResult
            The same result that was written to file.
        """
        result = self.crystallize(**kwargs)
        result.to_cif(filename)
        return result
