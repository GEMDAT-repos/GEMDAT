from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from gemdat.plots._shared import (
    _crystallizer_site_images,
    _crystallizer_sites,
    density_matching_colorscale_plotly,
)

from ._plot3d import plot_3d

if TYPE_CHECKING:
    from gemdat.crystallizer import CrystallizerScan


def crystallizer_sites(
    *,
    scan: CrystallizerScan,
    symprec: float | None = None,
    specie: str | None = None,
) -> go.Figure:
    """Plot the idealised sites of one species of a symmetry fit in 3D, inside
    the density of their species.

    The density is drawn as isosurfaces, see [gemdat.plots.plot_3d][]. Sites
    are coloured by orbit (symmetry-equivalent set) and labelled with their
    Wyckoff letter and multiplicity; hover shows the occupancy.

    Parameters
    ----------
    scan : CrystallizerScan
        Scan from [gemdat.crystallizer.Crystallizer.scan][].
    symprec : float | None
        Tolerance (Ångstrom) of the fit to plot, e.g. `result.symprec` or
        `level.symprec`. Defaults to the best (highest) space group of the scan.
    specie : str | None
        Species whose sites and density to plot, defaults to the mobile species.
        A framework species' sites are its time-averaged atom positions.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Output figure
    """
    title, sites, density = _crystallizer_sites(scan, symprec, specie)
    return plot_3d(volume=density, structure=sites, title=title)


def crystallizer_site_images(
    *,
    scan: CrystallizerScan,
    symprec: float | None = None,
    specie: str | None = None,
    site: int | None = None,
    width: float = 1.5,
    resolution: float = 0.05,
) -> go.Figure:
    """Plot one ideal site, the images the symmetry operations map onto it, and
    the density around it.

    The images are the density-derived sites of this site's orbit, each mapped
    onto it by one symmetry operation (the identity gives the site itself). The
    ideal position is (up to a common origin shift) their mean, see
    [gemdat.symmetry.SymmetryAnalyzer.idealise][], so their scatter shows
    how well this site obeys the symmetry. Three Cartesian views (xy, xz, yz)
    are centred on the ideal site, each over the density averaged along the
    viewing direction within the plotted box.

    Parameters
    ----------
    scan : CrystallizerScan
        Scan from [gemdat.crystallizer.Crystallizer.scan][].
    symprec : float | None
        Tolerance (Ångstrom) of the fit to plot. Defaults to the best (highest)
        space group of the scan.
    specie : str | None
        Species of the site to plot when `site` is not given, defaults to the
        mobile species.
    site : int | None
        Index of the site in `scan.geometry`. Defaults to the first site of
        `specie`. The background is the density of the site's species.
    width : float
        Half-width (Ångstrom) of the plotted box around the ideal site.
    resolution : float
        Spacing (Ångstrom) at which the density is sampled for the background.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Output figure
    """
    title, ticks, box, images = _crystallizer_site_images(
        scan, symprec, specie, site, width, resolution
    )

    views = (2, 1, 0)
    projections = {axis: box.mean(axis=axis).T for axis in views}
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            ''.join('xyz'[i] for i in range(3) if i != axis) + ' view' for axis in views
        ],
        horizontal_spacing=0.1,
    )
    # No fixed width, so the figure follows the notebook; `constrain='domain'`
    # keeps equal aspect by shrinking a panel instead of widening its range.
    fig.update_xaxes(constrain='domain', title_font_size=12, tickfont_size=10)
    fig.update_yaxes(title_font_size=12, tickfont_size=10, title_standoff=4)
    fig.update_annotations(font_size=13)  # subplot titles
    for col in (1, 2, 3):
        fig.update_yaxes(
            scaleanchor=f'x{col}' if col > 1 else 'x', constrain='domain', row=1, col=col
        )
    fig.update_layout(
        height=340,
        margin={'l': 50, 'r': 10, 't': 70, 'b': 45},
        plot_bgcolor='white',
        title_font_size=15,
        legend={'orientation': 'h', 'x': 1, 'xanchor': 'right', 'y': 1.2, 'yanchor': 'bottom'},
    )
    for col, axis in enumerate(views, start=1):
        x, y = (i for i in range(3) if i != axis)
        # The density itself, down to the faintest, with contour lines on top.
        fig.add_trace(
            go.Heatmap(
                x=ticks,
                y=ticks,
                z=projections[axis],
                coloraxis='coloraxis',
                zsmooth='best',
                hovertemplate='%{z:.3g} counts / voxel<extra></extra>',
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Contour(
                x=ticks,
                y=ticks,
                z=projections[axis],
                contours_coloring='none',
                line={'color': 'rgba(0,0,0,0.35)', 'width': 1},
                ncontours=8,
                showscale=False,
                showlegend=False,
                hoverinfo='skip',
            ),
            row=1,
            col=col,
        )
        # Images after the star, so a tight cluster is not hidden behind it.
        for name, points, marker in (
            ('ideal', np.zeros((1, 3)), {'color': '#d62728', 'size': 11, 'symbol': 'star'}),
            (
                'images',
                images,
                {'color': '#1f77b4', 'size': 6, 'line': {'width': 1, 'color': 'white'}},
            ),
        ):
            fig.add_trace(
                go.Scatter(
                    x=points[:, x],
                    y=points[:, y],
                    mode='markers',
                    marker=marker,
                    name=name,
                    legendgroup=name,
                    showlegend=col == 1,
                ),
                row=1,
                col=col,
            )
        fig.update_xaxes(title_text=f'{"xyz"[x]} (Å)', range=(-width, width), row=1, col=col)
        fig.update_yaxes(title_text=f'{"xyz"[y]} (Å)', range=(-width, width), row=1, col=col)

    fig.update_layout(
        title=title,
        coloraxis={
            'colorscale': density_matching_colorscale_plotly(),
            'cmin': 0,
            'cmax': max(projection.max() for projection in projections.values()),
            'colorbar': {
                'title': {'text': 'counts / voxel', 'side': 'right', 'font': {'size': 12}},
                'tickfont': {'size': 10},
                'thickness': 12,
            },
        },
    )

    return fig
