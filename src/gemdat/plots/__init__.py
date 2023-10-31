"""This module contains all the plots that Gemdat can generate."""
from __future__ import annotations

from ._density import density
from ._displacements import (
    displacement_histogram,
    displacement_histogram2,
    displacement_per_element,
    displacement_per_element2,
    displacement_per_site,
)
from ._jumps import (
    collective_jumps,
    jumps_3d,
    jumps_3d_animation,
    jumps_vs_distance,
    jumps_vs_distance2,
    jumps_vs_time,
    jumps_vs_time2,
)
from ._rdf import radial_distribution
from ._vibration import (
    frequency_vs_occurence,
    vibrational_amplitudes,
)

__all__ = [
    'density',
    'displacement_per_site',
    'displacement_per_element',
    'displacement_per_element2',
    'displacement_histogram',
    'displacement_histogram2',
    'frequency_vs_occurence',
    'vibrational_amplitudes',
    'jumps_vs_distance',
    'jumps_vs_distance2',
    'jumps_vs_time',
    'jumps_vs_time2',
    'collective_jumps',
    'jumps_3d',
    'jumps_3d_animation',
    'radial_distribution',
]
