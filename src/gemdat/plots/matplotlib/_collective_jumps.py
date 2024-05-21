from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from gemdat import Jumps


def collective_jumps(*, jumps: Jumps) -> plt.Figure:
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
    matrix = jumps.collective().site_pair_count_matrix()
    labels = jumps.collective().site_pair_count_matrix_labels()

    mat = ax.imshow(matrix)

    ticks = range(len(labels))

    ax.set_xticks(ticks, labels=labels, rotation=90)
    ax.set_yticks(ticks, labels=labels)

    fig.colorbar(mat, ax=ax)

    ax.set(title='Cooperative jumps per jump-type combination')

    return fig
