"""This module contains all the plots that Gemdat can generate."""
from __future__ import annotations

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
from .plotly._paths import (
    energy_along_path,
    path_on_grid,
    path_on_landscape,
)
from .plotly._vibration import vibrational_amplitudes

__all__ = [
    'density',
    'displacement_per_site',
    'displacement_per_element',
    'displacement_histogram',
    'frequency_vs_occurence',
    'vibrational_amplitudes',
    'jumps_vs_distance',
    'jumps_vs_time',
    'collective_jumps',
    'jumps_3d',
    'jumps_3d_animation',
    'radial_distribution',
    'energy_along_path',
    'path_on_grid',
    'path_on_landscape',
]
