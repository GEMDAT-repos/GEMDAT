"""This module contains all the plots that Gemdat can generate with plotly."""
from __future__ import annotations

from ._density import density
from ._displacement_histogram import displacement_histogram
from ._displacement_per_atom import displacement_per_atom
from ._displacement_per_element import displacement_per_element
from ._msd_per_element import msd_per_element
from ._collective_jumps import collective_jumps
from ._jumps_3d import jumps_3d
from ._jumps_vs_distance import jumps_vs_distance
from ._jumps_vs_time import jumps_vs_time
from ._plot3d import plot_3d
from ._frequency_vs_occurence import frequency_vs_occurence
from ._vibrational_amplitudes import vibrational_amplitudes

__all__ = [
    'collective_jumps',
    'density',
    'displacement_histogram',
    'displacement_per_atom',
    'displacement_per_element',
    'frequency_vs_occurence',
    'jumps_3d',
    'jumps_vs_distance',
    'jumps_vs_time',
    'plot_3d',
    'msd_per_element',
    'vibrational_amplitudes',
]
