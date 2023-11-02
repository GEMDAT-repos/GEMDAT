from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymatgen.core import Lattice, PeriodicSite, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.structure import SymmetrizedStructure

from .trajectory import Trajectory
from .utils import warn_lattice_not_close


@dataclass
class ShapeData:
    """Data class for storing shape data.

    Parameters
    ----------
    name : str
        Name or label for associated with this shape
    coords : np.ndarray
        (n, 3) coordinate array in cartesian system (Å)
    """
    name: str
    coords: np.ndarray

    def distances(self) -> np.ndarray:
        """Return distances from origin in Å."""
        return np.linalg.norm(self.coords, axis=1)

    @property
    def x(self):
        """Return x coordinates."""
        return self.coords[:, 0]

    @property
    def y(self):
        """Return y coordinates."""
        return self.coords[:, 1]

    @property
    def z(self):
        """Return z coordinates."""
        return self.coords[:, 2]


class ShapeAnalyzer:
    """The goal for this class is to have a generalized algorithm that for all
    symmetrically equivalent cluster centers, finds nearest atoms from
    trajectory or other list of positions, and transforms them back to the
    asymmetric unit.

    Combining symmetrically equivalent coordinates helps the statistics
    for performing shape analysis.
    """

    def __init__(self, symmetrized_structure: SymmetrizedStructure):
        """Takes a symmetrized structure to set up the site analyzer.

        Parameters
        ----------
        symmetrized_structure : SymmetrizedStructure
            Input structure, must be symmetrized.
        """
        self.unique_sites = [
            sites[0] for sites in symmetrized_structure.equivalent_sites
        ]
        self.lattice = symmetrized_structure.lattice
        self.symops = symmetrized_structure.spacegroup

    @classmethod
    def from_structure(cls, structure: Structure):
        """Contstruct instance from [Structure][pymatgen.core.Structure].

        The input structure is symmetrized using
        [SpacegroupAnalyzer][pymatgen.symmetry.analyzer.SpacegroupAnalyzer].

        Parameters
        ----------
        structure : Structure
            Input structure
        """
        sga = SpacegroupAnalyzer(structure)
        symmetrized_structure = sga.get_symmetrized_structure()
        return cls(symmetrized_structure=symmetrized_structure)

    def find_equivalent_positions(self,
                                  *,
                                  site: PeriodicSite,
                                  positions: np.ndarray,
                                  threshold: float = 1.0) -> np.ndarray:
        """Cluster all symmetrically equivalent positions within sphere around
        `site`.

        All equivalent positions are transformed back to the identity symmetry
        operation.

        Algorithm:
        - For every symmetry operation
            - Apply next symmetry operation to site coords
            - Find all positions within threshold distance
            - Copy and map points back to asymmetric unit (reverse symmetry op)
            - Subtract site coords (center on site)

        Parameters
        ----------
        site : PeriodicSite
            This site acts as the cluster center.
        positions : np.ndarray
            Positions to sample from.
        threshold : float, optional
            Cluster symmetrically equivalent positions
            within this distance from the given `site`.

        Returns
        -------
        centered : np.ndarray
            Clustered positions centered on `site` in Cartesian coordinate system
        """
        lattice = self.lattice
        symops = self.symops
        site_coords = site.frac_coords
        cluster = []

        for op in symops:
            sym_coords = op.operate(site_coords)
            dists = lattice.get_all_distances(sym_coords, positions)

            sel = dists < threshold
            close = positions[sel.flatten()]

            # digitize differences to move all close positions to
            # same sphere around coordr
            offsets = np.digitize(close - sym_coords, bins=[0.5, -0.4999999
                                                            ]) - 1
            close += offsets

            inversed = op.inverse.operate_multi(close)

            cluster.append(inversed)

        centered = np.vstack(cluster) - site_coords

        # convert to cartesian
        cart_coords = self.lattice.get_cartesian_coords(centered)
        return cart_coords

    def analyze_trajectory(self,
                           trajectory: Trajectory,
                           *,
                           supercell: None
                           | tuple[float, float, float] = None,
                           threshold: float = 1.0) -> list[ShapeData]:
        """Perform shape analysis on trajectory.

        Similar to [analyze_positions()][ShapeAnalyzer.analyze_positions]. Handles
        coordinate conversion it trajectory is a supercell of the structure used to
        instantiate this class. The trajectory lattice must be similar or a
        supercell thereof.

        Parameters
        ----------
        trajectory : Trajectory
            Input trajectory.
        supercell : None | tuple[float, float, float], optional
            If the trajectory is in a supercell of the input structure,
            the given supercell used to fold trajectory positions into same
            lattice.
        threshold : float, optional
            Cluster symmetrically equivalent positions
            within this distance from [unique_sites][ShapeAnalyzer.unique_sites].

        Returns
        -------
        shapes : list[ShapeData]
            Output shapes
        """
        test_lattice = trajectory.get_lattice(0)
        positions = trajectory.positions.reshape(-1, 3)

        if supercell is not None:
            scale_arr = np.array(supercell)

            scale_matrix = (1 / scale_arr) * np.eye(3)
            test_lattice = Lattice(np.dot(scale_matrix, test_lattice.matrix))

            positions = np.mod(positions, 1 / scale_arr) * scale_arr

        warn_lattice_not_close(self.lattice, test_lattice)

        return self.analyze_positions(positions=positions, threshold=threshold)

    def analyze_positions(self,
                          positions: np.ndarray,
                          *,
                          threshold: float = 1.0) -> list[ShapeData]:
        """Perform shape analysis on array of fractional coordinates.

        Parameters
        ----------
        positions : np.ndarray
            (n, 3) input array fractional coordinates. These must correspond to
            the same lattice as the structure used to instantiate this class.
        supercell : None | tuple[float, float, float], optional
            If the trajectory is in a supercell of the input structure,
            the given supercell used to fold trajectory positions into same
            lattice.
        threshold : float, optional
            Cluster symmetrically equivalent positions
            within this distance from [unique_sites][ShapeAnalyzer.unique_sites].

        Returns
        -------
        shapes : list[ShapeData]
            Output shapes
        """
        shapes = []

        for site in self.unique_sites:
            eqv_coords = self.find_equivalent_positions(site=site,
                                                        positions=positions,
                                                        threshold=threshold)

            shapes.append(ShapeData(name=site.label, coords=eqv_coords))

        return shapes
