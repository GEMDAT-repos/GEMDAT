from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Collection, Sequence

import numpy as np
from pymatgen.core import Lattice, PeriodicSite, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from ._plot_backend import plot_backend
from .trajectory import Trajectory
from .utils import warn_lattice_not_close

if TYPE_CHECKING:
    from pymatgen.symmetry.analyzer import SpacegroupOperations
    from pymatgen.symmetry.groups import SpaceGroup
    from pymatgen.symmetry.structure import SymmetrizedStructure


@dataclass
class ShapeData:
    """Data class for storing shape data.

    Parameters
    ----------
    name : str
        Name or label for associated with this shape
    coords : np.ndarray
        (n, 3) coordinate array in cartesian system (Å)
    radius : float
        Maximum distance from center (Å)
    """

    site: PeriodicSite
    coords: np.ndarray
    radius: float

    def distances(self) -> np.ndarray:
        """Return distances from origin in Å."""
        return np.linalg.norm(self.coords, axis=1)

    @property
    def origin(self) -> np.ndarray:
        """Return origin coordinates for site."""
        return self.site.coords

    @property
    def name(self) -> str:
        """Return name of shape."""
        return self.site.label

    @property
    def x(self) -> np.ndarray:
        """Return x coordinates."""
        return self.coords[:, 0]

    @property
    def y(self) -> np.ndarray:
        """Return y coordinates."""
        return self.coords[:, 1]

    @property
    def z(self) -> np.ndarray:
        """Return z coordinates."""
        return self.coords[:, 2]

    def centroid(self) -> np.ndarray:
        """Find centroid (center of mass) for given set of coordinates.

        Returns
        -------
        centroid : np.ndarray
            Center of coordinates.
        """
        centroid = np.mean(self.coords, axis=0)
        return centroid

    @plot_backend
    def plot(self, *, module, **kwargs):
        """See [gemdat.plots.shape][] for more info."""
        return module.shape(shape=self, **kwargs)


class ShapeAnalyzer:
    """The goal for this class is to have a generalized algorithm that for all
    symmetrically equivalent cluster centers, finds nearest atoms from
    trajectory or other list of positions, and transforms them back to the
    asymmetric unit.

    Combining symmetrically equivalent coordinates helps the statistics
    for performing shape analysis.
    """

    def __init__(
        self,
        *,
        sites: Collection[PeriodicSite],
        lattice: Lattice,
        spacegroup: SpaceGroup | SpacegroupOperations,
    ):
        """Set up shape analyzer from a collection of unique periodic sites,
        the lattice, and spacegroup.

        Parameters
        ----------
        sites : Collection[PeriodicSite]
            Collection of sites to cluster around
        lattice : Lattice
            Input lattice
        spacegroup : SpaceGroup
            Input spacegroup
        """
        self.sites = sites
        self.lattice = lattice
        self.spacegroup = spacegroup

    def __repr__(self):
        def to_str(x):
            return f'{x:>10.6f}'

        try:
            # account for api mismatch in SpaceGroup and SpacegroupOperations
            symbol = self.spacegroup.int_symbol
        except AttributeError:
            symbol = self.spacegroup.symbol

        out = [
            self.__class__.__name__,
            'Spacegroup',
            f'    {symbol} ({self.spacegroup.int_number})',
            'Lattice',
            f'    abc   : {" ".join(to_str(val) for val in self.lattice.abc)}',
            f'    angles: {" ".join(to_str(val) for val in self.lattice.angles)}',
            f'Unique sites ({len(self.sites)})',
        ]
        for site in self.sites:
            out.append(f'    {site!r}')

        return '\n    '.join(out)

    @classmethod
    def from_symmetrized_structure(cls, structure: SymmetrizedStructure):
        """Construct instance from [SymmetrizedStructure][pymatgen.symmetry.str
        ucture.SymmetrizedStructure].

        The input structure is already symmetrized using
        [SpacegroupAnalyzer][pymatgen.symmetry.analyzer.SpacegroupAnalyzer].

        Parameters
        ----------
        structure : SymmetrizedStructure
            Input symmetrized structure structure
        """
        unique_sites = [sites[0] for sites in structure.equivalent_sites]
        lattice = structure.lattice
        spacegroup = structure.spacegroup
        return cls(sites=unique_sites, lattice=lattice, spacegroup=spacegroup)

    @classmethod
    def from_structure(cls, structure: Structure):
        """Construct instance from [Structure][pymatgen.core.Structure].

        The input structure will be symmetrized using
        [SpacegroupAnalyzer][pymatgen.symmetry.analyzer.SpacegroupAnalyzer].

        Parameters
        ----------
        structure : Structure
            Input structure
        """
        sga = SpacegroupAnalyzer(structure)
        symmetrized_structure = sga.get_symmetrized_structure()
        return cls.from_symmetrized_structure(structure=symmetrized_structure)

    def find_equivalent_positions(
        self, *, site: PeriodicSite, positions: np.ndarray, radius: float = 1.0
    ) -> np.ndarray:
        """Cluster all symmetrically equivalent positions within sphere around
        `site`.

        All equivalent positions are transformed back to the identity symmetry
        operation.

        Algorithm:
        - For every symmetry operation
            - Apply next symmetry operation to site coords
            - Find all positions within threshold radius
            - Copy and map points back to asymmetric unit (reverse symmetry op)
            - Subtract site coords (center on site)

        Parameters
        ----------
        site : PeriodicSite
            This site acts as the cluster center.
        positions : np.ndarray
            Positions to sample from.
        radius : float, optional
            Cluster symmetrically equivalent positions
            within this distance from the given `site`.

        Returns
        -------
        centered : np.ndarray
            Clustered positions centered on `site` in Cartesian coordinate system
        """
        lattice = self.lattice
        spacegroup = self.spacegroup
        site_coords = site.frac_coords
        cluster = []

        for op in spacegroup:
            sym_coords = op.operate(site_coords)
            dists = lattice.get_all_distances(sym_coords, positions)

            sel = dists < radius
            close = positions[sel.flatten()]

            # digitize differences to move all close positions to
            # same sphere around coordr
            offsets = np.digitize(close - sym_coords, bins=[0.5, -0.4999999]) - 1
            close += offsets

            inversed = op.inverse.operate_multi(close)

            cluster.append(inversed)

        centered = np.vstack(cluster) - site_coords

        # convert to cartesian
        cart_coords = self.lattice.get_cartesian_coords(centered)
        return cart_coords

    def analyze_trajectory(
        self,
        trajectory: Trajectory,
        *,
        supercell: None | tuple[float, float, float] = None,
        radius: float = 1.0,
    ) -> list[ShapeData]:
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
        radius : float, optional
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

        return self.analyze_positions(positions=positions, radius=radius)

    def analyze_positions(
        self, positions: np.ndarray, *, radius: float = 1.0
    ) -> list[ShapeData]:
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
        radius : float, optional
            Cluster symmetrically equivalent positions
            within this distance from [unique_sites][ShapeAnalyzer.unique_sites].

        Returns
        -------
        shapes : list[ShapeData]
            Output shapes
        """
        shapes = []

        for site in self.sites:
            eqv_coords = self.find_equivalent_positions(
                site=site, positions=positions, radius=radius
            )

            shape = ShapeData(
                site=site,
                coords=eqv_coords,
                radius=radius,
            )

            shapes.append(shape)

        return shapes

    def shift_sites(
        self,
        vectors: Sequence[None | Sequence[float]],
        coords_are_cartesian: bool = True,
    ) -> ShapeAnalyzer:
        """Shift `.unique_sites` by given vectors.

        Parameters
        ----------
        vectors : Sequence[None | Sequence[float, float, float]]
            List of vectors matching the sites.
            If None, do not shift this site.
        coords_are_cartesian : bool, optional
            It true, vectors are cartesian, if false, fractional.

        Returns
        -------
        shape_analyzer : ShapeAnalyzer
            Shape analyzer with shifted sites.
        """
        new_sites = []

        for site, offset in zip(self.sites, vectors):
            if offset is None:
                new_sites.append(site)
                continue

            coords = site.coords if coords_are_cartesian else site.frac_coords

            new_site = PeriodicSite(
                site.species,
                coords + offset,
                self.lattice,
                coords_are_cartesian=coords_are_cartesian,
                label=site.label,
            )
            new_sites.append(new_site)

        return ShapeAnalyzer(sites=new_sites, spacegroup=self.spacegroup, lattice=self.lattice)

    def optimize_sites(
        self, shapes: Sequence[ShapeData], func: None | Callable = None
    ) -> ShapeAnalyzer:
        """Optimize unique sites from shape objects.

        Note: This function does not take into account special positions.

        Parameters
        ----------
        shapes : Sequence[Shape]
            List of input shapes. These must match `self.unique_sites`
        func : None | Callable, optional
            Specify function to calculate offsets in Cartesian setting.
            Must take `Shape` as its only argument.
            If None, use `Shape.centroid()` to determine offsets.

        Returns
        -------
        shape_analyzer : ShapeAnalyzer
            Shape analyzer with optimized sites.
        """
        vectors = []

        for shape in shapes:
            if func:
                vector = func(shape)
            else:
                vector = shape.centroid()

            vectors.append(vector)

        return self.shift_sites(vectors=vectors, coords_are_cartesian=True)

    def to_structure(self) -> Structure:
        """Retrieve structure from this object.

        Returns
        -------
        structure : Structure
            Pymatgen structure object
        """
        structure = Structure.from_spacegroup(
            sg=self.spacegroup.int_number,
            lattice=self.lattice,
            species=[site.specie for site in self.sites],
            coords=[site.frac_coords for site in self.sites],  # type: ignore
            labels=[site.label for site in self.sites],
        )
        return structure
