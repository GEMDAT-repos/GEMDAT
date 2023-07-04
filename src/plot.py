from typing import List, Optional, Union

import gemdat.plots as available_plots
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .data import SimulationData


def plot(plots: Union[List[str], str],
         data: Optional[SimulationData] = None,
         show: bool = True,
         **kwargs) -> List[Figure]:
    """Main plotting function of gemdat. it takes two mandatory arguments:

    - plots, a list of plot names, or just a plot name for the plot you want.
    - Optional arguments which are passed down to the plotting functions.

    Parameters
    ----------
    data : SimulationData
        data
    plots : Union[List[str],str]
        plots
    kwargs :
        kwargs

    Returns
    -------
    Figure:
        A list of matplotlib figures
    """

    # Convert plots to list, if it is not already a list
    if not isinstance(plots, list):
        plots = [plots]

    # extract data if present, but prioritise kwargs
    if data:
        kwargs = {**data.dict, **kwargs}

    figures = []

    for plot in plots:
        plot_function = getattr(available_plots, plot)
        figure = plot_function(**kwargs)
        figures.append(figure)

    if show:
        plt.show()

    return figures


def plot_all(**kwargs) -> List[Figure]:
    """The Plot All function finds out which plots are available for plotting,
    and plots those.

    Parameters
    ----------
    data : SimulationData
        data
    kwargs :
        kwargs

    Returns
    -------
    Figure:
        A list of matplotlib figures
    """
    return plot(plots=available_plots.__all__, **kwargs)
