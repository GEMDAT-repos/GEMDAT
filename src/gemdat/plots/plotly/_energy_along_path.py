from __future__ import annotations

import plotly.graph_objects as go
from pymatgen.core import Structure

from gemdat.path import Pathway


def energy_along_path(
    path: Pathway,
    *,
    structure: Structure,
    other_paths: list[Pathway] | None = None,
) -> go.Figure:
    """Plot energy along specified path.

    Parameters
    ----------
    path : Pathway
        Pathway object containing the energy along the path
    structure : Structure
        Structure object to get the site information
    other_paths : Pathway | list[Pathway]
        Optional list of alternative paths to plot

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Output figure
    """
    raise NotImplementedError
