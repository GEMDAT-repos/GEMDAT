from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Collection, Sequence

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import matplotlib.figure
    from matplotlib.colors import Colormap
    from pymatgem.core import PeriodicSite

    from gemdat.shape import ShapeData


def shape(
    shape: ShapeData,
    bins: int | Sequence[float] = 50,
    sites: Collection[PeriodicSite] | None = None,
    cmap: Colormap | None = None,
) -> matplotlib.figure.Figure:
    """Plot site cluster shapes.

    Parameters
    ----------
    shape : ShapeData
        Shape data to plot
    bins : int | Sequence[float]
        Number of bins or sequence of bin edges.
        See [hist()][matplotlib.pyplot.hist] for more info.
    sites : Collection[PeriodicSite] | None
        Plot these sites on the shape density

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    x_labels = ('X / Å', 'Y / Å', 'Z / Å')
    y_labels = ('Y / Å', 'Z / Å', 'X / Å')

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        sharex=True,
        figsize=(12, 5),
        gridspec_kw={'height_ratios': (4, 1)},
    )

    distances = shape.distances()

    R = np.mean(distances)
    std = np.std(distances)
    title = f'{shape.name}: R = {R:.3f}$~Å$, std = {std:.3f}'

    mean_dist = np.mean(distances)

    _tmp_dict = defaultdict(list)
    vector_dict = {}
    if sites:
        for site in sites:
            _tmp_dict[site.label].append(site.coords)
        for key, values in _tmp_dict.items():
            vector_dict[key] = np.array(values) - shape.origin

    coords = shape.coords

    axes[0, 1].set_title(title)  # type: ignore

    for col, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
        ax0 = axes[0, col]  # type: ignore
        ax1 = axes[1, col]  # type: ignore

        x_coords = coords[:, i]
        y_coords = coords[:, j]

        ax0.hist2d(x=x_coords, y=y_coords, bins=bins, cmap=cmap)
        ax0.set_ylabel(y_labels[col])

        circle = plt.Circle(
            (0, 0),
            mean_dist,
            color='r',
            linestyle='--',
            fill=False,
        )
        ax0.add_patch(circle)

        ax0.scatter(x=[0], y=[0], color='r', marker='.')

        for label, vects in vector_dict.items():
            x_vs = vects[:, i]
            y_vs = vects[:, j]

            for x, y in zip(x_vs, y_vs):
                ax0.text(x, y, s=label, color='r')

            ax0.scatter(x=x_vs, y=y_vs, color='r', marker='.', label=label)

        ax0.axis('equal')

        ax1.hist(x=x_coords, bins=bins, density=True)
        ax1.set_xlabel(x_labels[col])
        ax1.set_ylabel('density')

    fig.tight_layout()

    return fig
