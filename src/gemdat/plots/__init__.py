"""This module contains all the plots that Gemdat can generate."""
from __future__ import annotations

# Matplotlib plots
from .matplotlib import (
    collective_jumps,
    displacement_per_site,
    frequency_vs_occurence,
    jumps_3d,
    jumps_3d_animation,
    radial_distribution,
    shape,
)

# Plotly plots (matplotlib version might be available)
from .plotly import (
    density,
    displacement_histogram,
    displacement_per_element,
    jumps_vs_distance,
    jumps_vs_time,
    vibrational_amplitudes,
)

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
