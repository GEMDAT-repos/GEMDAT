from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymatgen.core import Lattice, PeriodicSite, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.structure import SymmetrizedStructure

from .trajectory import Trajectory
from .utils import warn_lattice_not_close


def plot_shape(shape: ShapeData, bins: int = 50):
    import matplotlib.pyplot as plt

    x_labels = ('X / Å', 'Y / Å', 'Z / Å')
    y_labels = ('Y / Å', 'Z / Å', 'X / Å')

    fig, axes = plt.subplots(nrows=2,
                             ncols=3,
                             sharex=True,
                             figsize=(12, 5),
                             gridspec_kw={'height_ratios': (4, 1)})
    fig.tight_layout()

    distances = np.sum(shape.coords**2, axis=1)**0.5

    msd = np.mean(distances**2)
    std = np.std(distances**2)
    title = f'{shape.name}: MSD = {msd:.3f}$~Å^2$, std = {std:.3f}'

    axes[0, 1].set_title(title)

    for i, (x, y) in enumerate(
        ((shape.x, shape.y), (shape.y, shape.z), (shape.z, shape.x))):
        ax0 = axes[0, i]
        ax1 = axes[1, i]

        ax0.hist2d(x=x, y=y, bins=bins)
        ax0.set_ylabel(y_labels[i])

        circle = plt.Circle((0, 0), msd, color='r', linestyle='--', fill=False)
        ax0.add_patch(circle)

        ax0.scatter(x=[0], y=[0], color='r', marker='.')
        ax0.axis('equal')

        ax1.hist(x=x, bins=bins, density=True)
        ax1.set_xlabel(x_labels[i])
        ax1.set_ylabel('density')

    fig.tight_layout()


@dataclass
class ShapeData:
    name: str
    coords: np.ndarray

    @property
    def x(self):
        return self.coords[:, 0]

    @property
    def y(self):
        return self.coords[:, 1]

    @property
    def z(self):
        return self.coords[:, 2]


class ShapeAnalyzer:

    def __init__(self, symmetrized_structure: SymmetrizedStructure):
        self.unique_sites = [
            sites[0] for sites in symmetrized_structure.equivalent_sites
        ]
        self.lattice = symmetrized_structure.lattice
        self.symops = symmetrized_structure.spacegroup

    @classmethod
    def from_structure(cls, structure: Structure):
        sga = SpacegroupAnalyzer(structure)
        symmetrized_structure = sga.get_symmetrized_structure()
        return cls(symmetrized_structure=symmetrized_structure)

    def _positions_from_trajectory(self,
                                   trajectory: Trajectory,
                                   *,
                                   supercell: None
                                   | tuple[float, float, float] = None):
        """Lattices must be similar.

        Supercell is used to fold trajectory positions into same
        lattice.
        """
        test_lattice = trajectory.get_lattice(0)
        positions = trajectory.positions.reshape(-1, 3)

        if supercell is not None:
            scale_arr = np.array(supercell)

            scale_matrix = (1 / scale_arr) * np.eye(3)
            test_lattice = Lattice(np.dot(scale_matrix, test_lattice.matrix))

            positions = np.mod(positions, 1 / scale_arr) * scale_arr

        warn_lattice_not_close(self.lattice, test_lattice)

        self.positions = positions

    def find_equivalent_positions(self,
                                  *,
                                  site: PeriodicSite,
                                  positions: np.ndarray,
                                  threshold: float = 1.0):
        lattice = self.lattice
        symops = self.symops
        unq_coord = site.frac_coords
        cluster = []

        for op in symops:
            cluster_coord = op.operate(unq_coord)
            dists = lattice.get_all_distances(cluster_coord, positions)

            sel = dists < threshold
            close = positions[sel.flatten()]

            # digitize differences to move all close positions to
            # same sphere around cluster center
            offsets = np.digitize(close - cluster_coord,
                                  bins=[0.5, -0.4999999]) - 1

            close += offsets

            inversed = op.inverse.operate_multi(close)

            print(f'size: {len(inversed)}', end=', ')
            print(
                'mean: {: .3f} {: .3f} {: .3f}'.format(*inversed.mean(axis=0)))

            cluster.append(inversed)

        centered = np.vstack(cluster) - unq_coord

        # convert to cartesian
        return self.lattice.get_cartesian_coords(centered)

    def analyze_trajectory(self,
                           *,
                           trajectory: Trajectory,
                           supercell: None
                           | tuple[float, float, float] = None,
                           threshold: float = 1.0) -> list[ShapeData]:
        """Lattices must be similar.

        Supercell is used to fold trajectory positions into same
        lattice.
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
                          *,
                          positions: np.ndarray,
                          threshold: float = 1.0) -> list[ShapeData]:
        shapes = []

        for site in self.unique_sites:
            print(f'\n{site.label}: {site.frac_coords}\n')

            eqv_coords = self.find_equivalent_positions(site=site,
                                                        positions=positions,
                                                        threshold=threshold)

            shapes.append(ShapeData(name=site.label, coords=eqv_coords))

        return shapes
