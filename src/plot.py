from typing import List, Optional, Union

import gemdat.plots as available_plots
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .trajectory import Trajectory


def plot(plots: Union[List[str], str],
         trajectory: Optional[Trajectory] = None,
         show: bool = True,
         **kwargs) -> List[Figure]:
    """Display all or a selection of plots.

    Parameters
    ----------
    plots : Union[List[str],str]
        List of plot names, or just a plot name for the plot you want.
    trajectory : Optional[Trajectory]
        Input trajectory
    show : bool
        Show plots if True
    kwargs : dict
        Optional arguments which are passed down to the plotting functions.

    Returns
    -------
    Figure:
        A list of matplotlib figures
    """
    if not isinstance(plots, list):
        plots = [plots]

    figures = []

    for plot in plots:
        plot_function = getattr(available_plots, plot)
        figure = plot_function(trajectory=trajectory, **kwargs)
        figures.append(figure)

    if show:
        plt.show()

    return figures


def plot_all(**kwargs) -> list[Figure]:
    """Display all available plots.

    Parameters
    ----------
    kwargs : dict
        Keyword arguments passed down to plotting functions

    Returns
    -------
    list[Figure]:
        A list of matplotlib figures
    """
    return plot(plots=available_plots.__all__, **kwargs)
