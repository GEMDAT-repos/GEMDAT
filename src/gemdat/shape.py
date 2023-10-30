from __future__ import annotations

import numpy as np
from pymatgen import symmetry
from pymatgen.core import Lattice, PeriodicSite, Structure

from .trajectory import Trajectory
from .utils import warn_lattice_not_close


def plot_cluster(coords: np.ndarray, name: str, bins: int = 50):
    import matplotlib.pyplot as plt

    axis_label = ('X / Å', 'Y / Å', 'Z / Å')

    fig, axes = plt.subplots(nrows=2,
                             ncols=3,
                             sharex=True,
                             figsize=(12, 5),
                             gridspec_kw={'height_ratios': (4, 1)})
    fig.tight_layout()

    distances = np.sum(coords**2, axis=1)**0.5

    msd = np.mean(distances**2)
    std = np.std(distances**2)
    title = f'{name}: MSD = {msd:.3f}$~Å^2$, std = {std:.3f}'

    axes[0, 1].set_title(title)

    for j, (axis_i, axis_j) in enumerate(((0, 1), (1, 2), (2, 0))):
        ax0 = axes[0, j]
        ax1 = axes[1, j]

        ax0.hist2d(x=coords[:, axis_i], y=coords[:, axis_j], bins=bins)
        ax0.set_ylabel(axis_label[axis_j])

        circle = plt.Circle((0, 0), msd, color='r', linestyle='--', fill=False)
        ax0.add_patch(circle)

        ax0.scatter(x=[0], y=[0], color='r', marker='.')
        ax0.axis('equal')

        ax1.hist(x=coords[:, axis_i], bins=bins, density=True)
        ax1.set_xlabel(axis_label[axis_i])
        ax1.set_ylabel('density')

    fig.tight_layout()


class ShapeAnalyzer:

    def __init__(self, structure: Structure):
        self.structure = structure
        self.sga = symmetry.analyzer.SpacegroupAnalyzer(structure)

        self.symops = self.sga.get_space_group_operations()
        symstruct = self.sga.get_symmetrized_structure()

        self.unique_sites = [sites[0] for sites in symstruct.equivalent_sites]
        self.lattice = symstruct.lattice

    def set_positions_from_trajectory(self,
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

    def _find_equivalent_positions(self,
                                   *,
                                   site: PeriodicSite,
                                   threshold: float = 1.0):
        lattice = self.lattice
        symops = self.symops
        positions = self.positions
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

        coords = np.vstack(cluster) - unq_coord

        # convert to cartesian
        return self.lattice.get_cartesian_coords(coords)

    def analyze(self, *, threshold: float = 1.0):
        clusters = {}

        for site in self.unique_sites:
            print(f'\n{site.label}: {site.frac_coords}\n')

            clusters[site.label] = self._find_equivalent_positions(
                site=site, threshold=threshold)

        return clusters
