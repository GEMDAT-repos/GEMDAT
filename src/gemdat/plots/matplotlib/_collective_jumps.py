from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import matplotlib.figure

    from gemdat import Jumps


def collective_jumps(*, jumps: Jumps) -> matplotlib.figure.Figure:
    """Plot collective jumps per jump-type combination.

    Parameters
    ----------
    jumps : Jumps
        Input data

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    fig, ax = plt.subplots()

    collective = jumps.collective()

    matrix = collective.site_pair_count_matrix()

    im = ax.imshow(matrix)

    labels = collective.site_pair_count_matrix_labels()
    ticks = range(len(labels))

    ax.set_xticks(ticks, labels=labels, rotation=90)
    ax.set_yticks(ticks, labels=labels)

    fig.colorbar(im, ax=ax)

    ax.set(title='Cooperative jumps per jump-type combination')

    return fig
