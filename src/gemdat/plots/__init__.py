"""This module contains all the plots that Gemdat can generate."""
from __future__ import annotations

from .matplotlib import shape

# Matplotlib plots
from .matplotlib._displacements import displacement_per_site
from .matplotlib._jumps import (
    collective_jumps,
    jumps_3d,
    jumps_3d_animation,
)
from .matplotlib._rdf import radial_distribution
from .matplotlib._vibration import frequency_vs_occurence

# Plotly plots (matplotlib version might be available)
from .plotly._density import density
from .plotly._displacements import (
    displacement_histogram,
    displacement_per_element,
)
from .plotly._jumps import (
    jumps_vs_distance,
    jumps_vs_time,
)
from .plotly._vibration import vibrational_amplitudes

__all__ = [
    'collective_jumps',
    'density',
    'displacement_histogram',
    'displacement_per_element',
    'displacement_per_site',
    'frequency_vs_occurence',
    'jumps_3d',
    'jumps_3d_animation',
    'jumps_vs_distance',
    'jumps_vs_time',
    'radial_distribution',
    'shape',
    'vibrational_amplitudes',
]
