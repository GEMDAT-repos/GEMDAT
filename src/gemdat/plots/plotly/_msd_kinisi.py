from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import plotly.graph_objects as go

from gemdat.plots._shared import hex2rgba

if TYPE_CHECKING:
    from kinisi.analyze import DiffusionAnalyzer


def msd_kinisi(
    trajectory: Trajectory,
    specie: str,
    *,
    diffusion_analyzer: Optional['DiffusionAnalyzer'] = None,
    step_skip: int = 1,
    dt: 'sc.Variable | None' = None,
    dimension: str = 'xyz',
    distance_unit: str = 'angstrom',
    specie_indices: 'sc.Variable | None' = None,
    masses: 'sc.Variable | None' = None,
    progress: bool = True,
    cache: bool = True,
    show_shaded: bool = True,
) -> go.Figure:
    """Plot mean-squared displacement (MSD) with uncertainties from a kinisi DiffusionAnalyzer.

    Parameters
    ----------
    trajectory
        GEMDAT trajectory
    specie
        Specie to calculate diffusivity for, e.g. ``"Li"``.
    diffusion_analyzer
        A kinisi DiffusionAnalyzer instance.
    step_skip
        Number of MD integrator time steps between stored frames.
    dt
        Time intervals to calculate displacements over. Optional; if ``None``,
        kinisi defaults to a regular grid from the smallest interval
        (``time_step * step_skip``) to the full trajectory length.
    dimension
        Subset of ``"xyz"`` indicating displacement axes of interest.
    distance_unit
        Unit of distance in the input structures, as a string understood by
        ``scipp.Unit(...)`` (default: ``"angstrom"``).
    specie_indices
        Indices of the specie to calculate the diffusivity for. Optional; if ``None``, kinisi selects
        indices based on ``specie``.
    masses
        Masses for centre-of-mass handling. Optional.
    progress
        Show progress bars during parsing and MSD evaluation.
    cache
        Cache the populated analyzer on this trajectory instance.
        Cached data can be accessed via ``trajectory.kinisi_diffusion_analyzer_cache``.
    show_shaded : bool, optional
        If True, plot ±1σ uncertainties as a shaded region.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Output figure.
    """
    if diffusion_analyzer:
        cache_data = diffusion_analyzer
    else:
        cache_data = getattr(trajectory, 'kinisi_diffusion_analyzer_cache', None)
    if cache_data is None:
        cache_data = trajectory.to_kinisi_diffusion_analyzer(
            specie=specie,
            step_skip=step_skip,
            dt=dt,
            dimension=dimension,
            distance_unit=distance_unit,
            specie_indices=specie_indices,
            masses=masses,
            progress=progress,
            cache=cache,
    )

    dt = cache_data.dt
    msd = cache_data.msd

    x = np.asarray(dt.values)
    y = np.asarray(msd.values)

    variances = cache_data.msd.variances
    yerr = None if variances is None else np.sqrt(np.asarray(variances))

    fig = go.Figure()

    color_hex = fig.layout['template']['layout']['colorway'][0]
    color_rgba = hex2rgba(color_hex, opacity=0.3)

    name = f'{specie} MSD'

    if (yerr is not None) and show_shaded:
        name = f'{specie} MSD ± 1σ'
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y + yerr,
                fillcolor=color_rgba,
                mode='lines',
                line={'width': 0},
                legendgroup=specie,
                showlegend=False,
                zorder=0,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y - yerr,
                fillcolor=color_rgba,
                mode='none',
                legendgroup=specie,
                fill='tonexty',
                showlegend=False,
                zorder=0,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            name=name,
            mode='lines',
            line={'width': 3, 'color': color_hex},
            legendgroup=specie,
            zorder=1,
        )
    )
    
    fig.update_layout(
        showlegend=True,
        title='Mean squared displacement',
        xaxis_title=f'Time lag ({dt.unit})',
        yaxis_title=f'MSD ({msd.unit})',
    )

    return fig
