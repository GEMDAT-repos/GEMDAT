"""This module contains all the plots that Gemdat can generate with
matplotlib."""
from __future__ import annotations

from ._displacements import (
    displacement_histogram,
    displacement_per_element,
    displacement_per_site,
)
from ._jumps import (
    collective_jumps,
    jumps_3d,
    jumps_3d_animation,
    jumps_vs_distance,
    jumps_vs_time,
)
from ._rdf import radial_distribution
from ._shape import shape
from ._vibration import (
    frequency_vs_occurence,
    vibrational_amplitudes,
)
from .matplotlib._paths import (
    energy_along_path,
    path_on_grid,
)

__all__ = [
    'collective_jumps',
    'displacement_histogram',
    'displacement_per_element',
    'displacement_per_site',
    'energy_along_path',
    'frequency_vs_occurence',
    'jumps_3d',
    'jumps_3d_animation',
    'jumps_vs_distance',
    'jumps_vs_time',
    'path_on_grid',
    'radial_distribution',
    'shape',
    'vibrational_amplitudes',
]
