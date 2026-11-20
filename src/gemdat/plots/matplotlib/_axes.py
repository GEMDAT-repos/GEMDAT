from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure


def get_axes(
    ax: matplotlib.axes.Axes | None,
    *,
    projection: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[matplotlib.figure.Figure | None, matplotlib.axes.Axes]:
    """Return `(None, ax)` for a user-supplied axes, else `(fig, ax)` of a new
    figure.

    Raises `TypeError` if `projection='3d'` and `ax` is not a 3D axes.
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        return fig, fig.add_subplot(projection=projection)
    if projection == '3d' and ax.name != '3d':
        raise TypeError(
            f'This plot needs a 3D axes, got {ax.name!r}; create one with '
            "`fig.add_subplot(projection='3d')`."
        )
    return None, ax
