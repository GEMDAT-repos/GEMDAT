"""This module contains all the plots that Gemdat can generate with
matplotlib."""
from __future__ import annotations

from ._displacements import (
    displacement_histogram,
    displacement_per_element,
    displacement_per_site,
    msd_per_element,
)
from ._jumps import (
    collective_jumps,
    jumps_3d,
    jumps_3d_animation,
    jumps_vs_distance,
    jumps_vs_time,
)
from ._paths import (
    energy_along_path,
    path_on_grid,
)
from ._rdf import radial_distribution
from ._rotations import rectilinear_plot
from ._shape import shape
from ._vibration import (
    frequency_vs_occurence,
    vibrational_amplitudes,
)

__all__ = [
    'collective_jumps',
    'displacement_histogram',
    'displacement_per_element',
    'displacement_per_site',
    'energy_along_path',
    'frequency_vs_occurence',
    'jumps_3d',
    'jumps_3d_animation',
    'jumps_vs_distance',
    'jumps_vs_time',
    'msd_per_element',
    'path_on_grid',
    'radial_distribution',
    'rectilinear_plot',
    'shape',
    'vibrational_amplitudes',
]
