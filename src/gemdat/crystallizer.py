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

if TYPE_CHECKING:
    from pathlib import Path

    from .trajectory import Trajectory


@dataclass
class CrystallizerResult:
    """Container for the result of [Crystallizer.crystallize][gemdat.crystallize
    r.Crystallizer.crystallize].

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
        Symmetry tolerance (in Ångstrom) that produced the highest-symmetry fit.
    """

    structure: Structure
    spacegroup_symbol: str
    spacegroup_number: int
    symprec: float


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
            [gemdat.Trajectory.to_volume][].
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
            [gemdat.Volume.to_structure][].
        **find_peaks_kwargs : dict
            Passed through to [gemdat.Volume.find_peaks][].

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

        The returned structure has every site at full occupancy (element symbols
        only). Symmetry must be searched on this structure: the per-site
        occupancies are continuous floats, and feeding them to the symmetry
        finder would make every mobile site distinct and collapse the result to
        P1. The occupancies are returned separately, aligned to the structure's
        site order (framework sites are 1.0), so they can be averaged over the
        symmetry-equivalent classes afterwards.
        """
        framework = self.framework()
        mobile = self.mobile_sites(background_level=background_level, **find_peaks_kwargs)

        symbols = [site.specie.symbol for site in framework]
        symbols += [next(iter(site.species.as_dict())) for site in mobile]

        occupancies = [1.0] * len(framework)
        occupancies += [site.species.num_atoms for site in mobile]

        coords = np.vstack([framework.frac_coords, mobile.frac_coords])

        geometry = Structure(
            lattice=framework.lattice,
            species=symbols,
            coords=coords,
        )
        return geometry, np.array(occupancies)

    def crystallize(
        self,
        *,
        symprec_range: tuple[float, ...] = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5),
        angle_tolerance: float = 5.0,
        background_level: float = 0.1,
        **find_peaks_kwargs,
    ) -> CrystallizerResult:
        """Build the full structure and fit the highest possible space group.

        Parameters
        ----------
        symprec_range : tuple[float, ...]
            Symmetry tolerances (Ångstrom) to try. The tolerance giving the
            highest space group number wins; ties are broken towards the
            tightest (smallest) tolerance.
        angle_tolerance : float
            Angle tolerance (degrees) passed to
            [pymatgen.symmetry.analyzer.SpacegroupAnalyzer][].
        background_level : float
            Fraction of the maximum density used as the segmentation floor.
        **find_peaks_kwargs : dict
            Passed through to [gemdat.Volume.find_peaks][].

        Returns
        -------
        result : CrystallizerResult
            The symmetrized structure and the fitted space group.
        """
        geometry, occupancies = self._geometry_and_occupancies(
            background_level=background_level, **find_peaks_kwargs
        )

        best_symprec = None
        best_number = 0
        for symprec in symprec_range:
            try:
                number = SpacegroupAnalyzer(
                    geometry, symprec=symprec, angle_tolerance=angle_tolerance
                ).get_space_group_number()
            except Exception:
                continue
            if number > best_number:
                best_number = number
                best_symprec = symprec

        if best_symprec is None:
            raise ValueError(
                'Could not determine symmetry for any of the given symprec values.'
            )

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
