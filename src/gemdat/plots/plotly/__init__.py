"""This module contains all the plots that Gemdat can generate with plotly."""

from __future__ import annotations

from ._arrhenius import arrhenius
from ._autocorrelation import autocorrelation
from ._bond_length_distribution import bond_length_distribution
from ._collective_jumps import collective_jumps
from ._density import density
from ._displacement_histogram import displacement_histogram
from ._displacement_per_atom import displacement_per_atom
from ._displacement_per_element import displacement_per_element
from ._energy_along_path import energy_along_path
from ._frequency_vs_occurence import frequency_vs_occurence
from ._jumps_3d import jumps_3d
from ._jumps_vs_distance import jumps_vs_distance
from ._jumps_vs_time import jumps_vs_time
from ._msd_kinisi import msd_kinisi
from ._msd_per_element import msd_per_element
from ._plot3d import plot_3d, plot_3d_points
from ._polar import polar
from ._radial_distribution import radial_distribution
from ._rectilinear import rectilinear
from ._shape import shape
from ._vibrational_amplitudes import vibrational_amplitudes

__all__ = [
    'arrhenius',
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
    'jumps_vs_distance',
    'jumps_vs_time',
    'msd_kinisi',
    'msd_per_element',
    'plot_3d',
    'plot_3d_points',
    'polar',
    'radial_distribution',
    'rectilinear',
    'shape',
    'vibrational_amplitudes',
]
