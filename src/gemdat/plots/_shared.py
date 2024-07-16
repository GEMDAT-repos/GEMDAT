from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import skewnorm

if TYPE_CHECKING:
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
