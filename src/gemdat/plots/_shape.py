from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from gemdat.shape import ShapeData


def shape(shape: ShapeData, bins: int | Sequence[float] = 50) -> plt.Figure:
    """Plot site cluster shapes.

    Parameters
    ----------
    shape : ShapeData
        Shape data to plot
    bins : int | Sequence[float]
        Number of bins or sequence of bin edges.
        See [hist()][matplotlib.pyplot.hist] for more info.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    x_labels = ('X / Å', 'Y / Å', 'Z / Å')
    y_labels = ('Y / Å', 'Z / Å', 'X / Å')

    fig, axes = plt.subplots(nrows=2,
                             ncols=3,
                             sharex=True,
                             figsize=(12, 5),
                             gridspec_kw={'height_ratios': (4, 1)})

    distances_sq = shape.distances()**2

    msd = np.mean(distances_sq)
    std = np.std(distances_sq)
    title = f'{shape.name}: MSD = {msd:.3f}$~Å^2$, std = {std:.3f}'

    axes[0, 1].set_title(title)

    for i, (x, y) in enumerate(
        ((shape.x, shape.y), (shape.y, shape.z), (shape.z, shape.x))):
        ax0 = axes[0, i]
        ax1 = axes[1, i]

        ax0.hist2d(x=x, y=y, bins=bins)
        ax0.set_ylabel(y_labels[i])

        circle = plt.Circle((0, 0), msd, color='r', linestyle='--', fill=False)
        ax0.add_patch(circle)

        ax0.scatter(x=[0], y=[0], color='r', marker='.')
        ax0.axis('equal')

        ax1.hist(x=x, bins=bins, density=True)
        ax1.set_xlabel(x_labels[i])
        ax1.set_ylabel('density')

    fig.tight_layout()
