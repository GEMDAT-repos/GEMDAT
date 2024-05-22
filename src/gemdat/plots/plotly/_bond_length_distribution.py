from __future__ import annotations

import plotly.graph_objects as go

from gemdat.orientations import Orientations


def bond_length_distribution(*,
                             orientations: Orientations,
                             bins: int = 1000) -> go.Figure:
    """Plot the bond length probability distribution.

    Parameters
    ----------
    orientations : Orientations
        The unit vector trajectories
    bins : int, optional
        The number of bins, by default 1000

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Output figure
    """
    raise NotImplementedError
