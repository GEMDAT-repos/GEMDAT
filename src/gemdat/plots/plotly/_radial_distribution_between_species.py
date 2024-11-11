from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

from .._shared import _get_radial_distribution_between_species

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
    bins, rdf = _get_radial_distribution_between_species(
        trajectory=trajectory,
        specie_1=specie_1,
        specie_2=specie_2,
        max_dist=max_dist,
        resolution=resolution,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bins,
            y=rdf,
            name='Radial distribution',
            mode='lines',
        )
    )
    str1 = specie_1 if isinstance(specie_1, str) else ' / '.join(specie_1)
    str2 = specie_1 if isinstance(specie_2, str) else ' / '.join(specie_2)
    fig.update_layout(
        title=f'RDF between {str1} and {str2}',
        xaxis_title='Radius (Å)',
        yaxis_title='Nr. of atoms',
    )
    return fig
