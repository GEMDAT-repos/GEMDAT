from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING
from collections import defaultdict
if TYPE_CHECKING:
    from gemdat.trajectory import Trajectory


def _mean_displacements_per_element(
        trajectory: Trajectory) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Calculate mean displacements per element type.

    Helper for `displacement_per_atom`.
    """
    species = trajectory.species

    grouped = defaultdict(list)
    for sp, distances in zip(species,
                             trajectory.distances_from_base_position()):
        grouped[sp.symbol].append(distances)

    means = {}
    for sp, dists in grouped.items():
        mean = np.mean(dists, axis=0)
        std = np.std(dists, axis=0)

        means[sp] = (mean, std)

    return means
