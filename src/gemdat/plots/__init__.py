"""This module contains all the plots that Gemdat can generate."""
from __future__ import annotations

# Matplotlib plots
from .matplotlib import (
    energy_along_path,
    jumps_3d_animation,
    path_on_grid,
    radial_distribution,
    shape,
)

# Plotly plots (matplotlib version might be available)
from .plotly import (
    collective_jumps,
    density,
    displacement_histogram,
    displacement_per_atom,
    displacement_per_element,
    frequency_vs_occurence,
    jumps_3d,
    jumps_vs_distance,
    jumps_vs_time,
    msd_per_element,
    plot_3d,
    vibrational_amplitudes,
)

__all__ = [
    'collective_jumps',
    'density',
    'displacement_histogram',
    'displacement_per_atom',
    'displacement_per_element',
    'frequency_vs_occurence',
    'jumps_3d',
    'jumps_3d_animation',
    'jumps_vs_distance',
    'jumps_vs_time',
    'msd_per_element',
    'plot_3d',
    'radial_distribution',
    'shape',
    'vibrational_amplitudes',
    'energy_along_path',
    'path_on_grid',
]
