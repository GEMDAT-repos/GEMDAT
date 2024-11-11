from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

if TYPE_CHECKING:
    from typing import Collection

    from gemdat import Trajectory


def radial_distribution_between_species(
    trajectory: Trajectory,
    specie_1: str | Collection[str],
    specie_2: str | Collection[str],
    max_dist: float = 5.0,
    resolution: float = 0.1,
) -> go.Figure:
    """Calculate RDFs from specie_1 to specie_2.

    Parameters
    ----------
    trajectory: Trajectory
        Input trajectory.
    specie_1: str | list[str]
        Name of specie or list of species
    specie_2: str | list[str]
        Name of specie or list of species
    max_dist: float, optional
        Max distance for rdf calculation
    resolution: float, optional
        Width of the bins

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    fig = go.Figure()
    raise NotImplementedError
