"""This module contains all the plots that Gemdat can generate with
matplotlib."""

from __future__ import annotations

from ._autocorrelation import autocorrelation
from ._bond_length_distribution import bond_length_distribution
from ._collective_jumps import collective_jumps
from ._displacement_histogram import displacement_histogram
from ._displacement_per_atom import displacement_per_atom
from ._displacement_per_element import displacement_per_element
from ._energy_along_path import energy_along_path
from ._frequency_vs_occurence import frequency_vs_occurence
from ._jumps_3d import jumps_3d
from ._jumps_3d_animation import jumps_3d_animation
from ._jumps_vs_distance import jumps_vs_distance
from ._jumps_vs_time import jumps_vs_time
from ._msd_per_element import msd_per_element
from ._polar import polar
from ._radial_distribution import radial_distribution
from ._rectilinear import rectilinear
from ._shape import shape
from ._vibrational_amplitudes import vibrational_amplitudes

__all__ = [
    'autocorrelation',
    'bond_length_distribution',
    'collective_jumps',
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
    'polar',
    'radial_distribution',
    'rectilinear',
    'shape',
    'vibrational_amplitudes',
]
