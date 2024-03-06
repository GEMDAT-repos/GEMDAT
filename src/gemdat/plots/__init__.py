"""This module contains all the plots that Gemdat can generate."""
from __future__ import annotations

# Matplotlib plots
from .matplotlib import (
    bond_length_distribution,
    collective_jumps,
    displacement_per_site,
    energy_along_path,
    frequency_vs_occurence,
    jumps_3d,
    jumps_3d_animation,
    path_on_grid,
    radial_distribution,
    rectilinear_plot,
    shape,
)

# Plotly plots (matplotlib version might be available)
from .plotly import (
    density,
    displacement_histogram,
    displacement_per_element,
    jumps_vs_distance,
    jumps_vs_time,
    msd_per_element,
    path_on_landscape,
    vibrational_amplitudes,
)

__all__ = [
    'bond_length_distribution',
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
    'msd_per_element',
    'radial_distribution',
    'rectilinear_plot',
    'shape',
    'vibrational_amplitudes',
    'energy_along_path',
    'path_on_grid',
    'path_on_landscape',
]
