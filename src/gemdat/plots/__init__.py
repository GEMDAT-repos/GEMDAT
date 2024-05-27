"""This module contains all the plots that Gemdat can generate."""
from __future__ import annotations

from .matplotlib import (
    autocorrelation,
    bond_length_distribution,
    energy_along_path,
    jumps_3d_animation,
    radial_distribution,
    rectilinear,
    shape,
)

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
    'autocorrelation',
    'bond_length_distribution',
    'collective_jumps',
    'density',
    'displacement_histogram',
    'displacement_per_atom',
    'displacement_per_element',
    'energy_along_path',
    'frequency_vs_occurence',
    'jumps_3d',
    'jumps_3d_animation',
    'jumps_vs_distance',
    'jumps_vs_time',
    'msd_per_element',
    'msd_per_element',
    'plot_3d',
    'radial_distribution',
    'rectilinear',
    'shape',
    'vibrational_amplitudes',
]
