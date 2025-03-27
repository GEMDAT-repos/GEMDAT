"""This module contains all the plots that Gemdat can generate."""

from __future__ import annotations

from .matplotlib import (
    jumps_3d_animation,
    polar,
)
from .plotly import (
    autocorrelation,
    bond_length_distribution,
    collective_jumps,
    density,
    displacement_histogram,
    displacement_per_atom,
    displacement_per_element,
    energy_along_path,
    frequency_vs_occurence,
    jumps_3d,
    jumps_vs_distance,
    jumps_vs_time,
    msd_per_element,
    plot_3d,
    radial_distribution,
    rectilinear,
    shape,
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
    'polar',
    'radial_distribution',
    'rectilinear',
    'shape',
    'vibrational_amplitudes',
]
