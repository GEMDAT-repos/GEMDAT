from typing import List, Optional, Union

import gemdat.plots as available_plots
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .trajectory import GemdatTrajectory


def plot(plots: Union[List[str], str],
         trajectory: Optional[GemdatTrajectory] = None,
         show: bool = True,
         **kwargs) -> List[Figure]:
    """Display all or a selection of plots.

    Parameters
    ----------
    trajectory : GemdatTrajectory
        Input trajectory
    plots : Union[List[str],str]
        List of plot names, or just a plot name for the plot you want.
    kwargs : dict
        Optional arguments which are passed down to the plotting functions.

    Returns
    -------
    Figure:
        A list of matplotlib figures
    """

    # Convert plots to list, if it is not already a list
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


def plot_all(**kwargs) -> List[Figure]:
    """The Plot All function finds out which plots are available for plotting,
    and plots those.

    Parameters
    ----------
    data : GemdatTrajectory
        data
    kwargs :
        kwargs

    Returns
    -------
    Figure:
        A list of matplotlib figures
    """
    return plot(plots=available_plots.__all__, **kwargs)
