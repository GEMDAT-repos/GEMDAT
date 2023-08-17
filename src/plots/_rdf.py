import matplotlib.pyplot as plt
import numpy as np


def radial_distribution(rdfs: dict[str, np.ndarray],
                        name: str | None = None) -> plt.Figure:
    """Plot radial distribution function.

    Parameters
    ----------
    rdfs : dict[str, np.ndarray]
        Dictionary with rdf array per symbol

    Returns
    -------
    fig : plt.Figure
        Output matplotlib figure
    """

    fig, ax = plt.subplots()

    for symbol, rdf in rdfs.items():
        ax.plot(rdf[:-1], label=symbol)

    suffix = f' ({name})' if name else ''

    ax.legend()
    ax.set(title=f'Radial distribution function{suffix}',
           xlabel='Distance (Ang)',
           ylabel='Counts')

    return fig
