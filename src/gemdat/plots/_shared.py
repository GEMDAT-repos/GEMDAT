from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pymatgen.core import Element, Species
from scipy.optimize import curve_fit
from scipy.stats import skewnorm

if TYPE_CHECKING:
    from gemdat.jumps import Jumps
    from gemdat.orientations import Orientations
    from gemdat.trajectory import Trajectory


def _mean_displacements_per_element(
    trajectory: Trajectory,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Calculate mean displacements per element type.

    Helper for `displacement_per_atom`.
    """
    species = trajectory.species

    grouped = defaultdict(list)
    for sp, distances in zip(species, trajectory.distances_from_base_position()):
        assert isinstance(sp, (Species, Element)), f'got {type(sp)}'

        grouped[sp.symbol].append(distances)

    means = {}
    for sp, dists in grouped.items():
        mean = np.mean(dists, axis=0)
        std = np.std(dists, axis=0)

        means[sp] = (mean, std)

    return means


def _orientations_to_histogram(
    orientations: Orientations, /, *, bins: int = 50
) -> pd.DataFrame:
    """Calculate bond lenth histogram from `Orientations`

    Helper for `bond_length_distribution`.
    """
    *_, bond_lengths = orientations.vectors_spherical.T
    bond_lengths = bond_lengths.flatten()

    hist, edges = np.histogram(bond_lengths, bins=bins, density=True)
    bin_centers = bin_centers = (edges[:-1] + edges[1:]) / 2

    return pd.DataFrame(
        {
            'prob': hist,
            'center': bin_centers,
            'left_edge': edges[:-1],
            'right_edge': edges[1:],
        }
    )


def _fit_skewnorm_to_hist(
    df: pd.DataFrame, /, steps: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Fit skewnorm to bond lenth histogram from `Orientations`

    Helper for `bond_length_distribution`.
    """
    bin_centers = df['center']
    hist = df['prob']

    params, covariance = curve_fit(skewnorm.pdf, bin_centers, hist, p0=[1.5, 1, 1.5])

    x = np.linspace(min(bin_centers), max(bin_centers), steps)
    y = skewnorm.pdf(x, *params)

    return x, y


def hex2rgba(hex_color: str, *, opacity: float = 1) -> str:
    """Convert hex string to rgba."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)

    return f'rgba({r},{g},{b},{opacity})'


@dataclass
class VibrationalAmplitudeHist:
    amplitudes: np.ndarray
    counts: np.ndarray
    std: np.ndarray

    @property
    def min_amp(self):
        return self.amplitudes.min()

    @property
    def max_amp(self):
        return self.amplitudes.max()

    @property
    def width(self):
        bins = len(self.amplitudes)
        return (self.max_amp - self.min_amp) / bins

    @property
    def offset(self):
        return self.width / 2

    @property
    def centers(self):
        return self.amplitudes + self.offset

    @property
    def dataframe(self):
        return pd.DataFrame(
            data=zip(self.centers, self.counts, self.std), columns=['center', 'count', 'std']
        )


def _get_vibrational_amplitudes_hist(
    *, trajectories: list[Trajectory], bins: int
) -> VibrationalAmplitudeHist:
    """Calculate vibrational amplitudes histogram.

    Helper for `vibrational_amplitudes`.
    """
    metrics = [trajectory.metrics().amplitudes() for trajectory in trajectories]

    max_amp = max(max(metric) for metric in metrics)
    min_amp = min(min(metric) for metric in metrics)

    max_amp = max(abs(min_amp), max_amp)
    min_amp = -max_amp

    data = []

    for metric in metrics:
        data.append(np.histogram(metric, bins=bins, range=(min_amp, max_amp), density=True)[0])

    amplitudes = np.linspace(min_amp, max_amp, bins, endpoint=False)

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    return VibrationalAmplitudeHist(amplitudes=amplitudes, counts=mean, std=std)


def _jumps_vs_distance(jumps: Jumps, *, resolution: float, n_parts: int) -> pd.DataFrame:
    """Calculate jumps vs distance histogram.

    Helper for `jumps_vs_distance`.
    """
    sites = jumps.sites
    trajectory = jumps.trajectory
    lattice = trajectory.get_lattice()

    pdist = lattice.get_all_distances(sites.frac_coords, sites.frac_coords)

    bin_max = (1 + pdist.max() // resolution) * resolution
    n_bins = int(bin_max / resolution) + 1
    x = np.linspace(0, bin_max, n_bins)

    bin_idx = np.digitize(pdist, bins=x)
    data = []
    for transitions_part in jumps.split(n_parts=n_parts):
        counts = np.zeros_like(x)
        for idx, n in zip(bin_idx.flatten(), transitions_part.matrix().flatten()):
            counts[idx] += n
        for idx in range(n_bins):
            if counts[idx] > 0:
                data.append((x[idx], counts[idx]))

    df = pd.DataFrame(data=data, columns=['Displacement', 'count'])

    grouped = df.groupby(['Displacement'])
    mean = grouped.mean().reset_index().rename(columns={'count': 'mean'})
    std = grouped.std().reset_index().rename(columns={'count': 'std'})
    df = mean.merge(std, how='inner')

    return df
