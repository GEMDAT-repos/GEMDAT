from typing import List, Union

import GEMDAT.plots as available_plots

from .data import Data


def plot(data: Data, plots: Union[List[str], str], **kwargs) -> None:
    """Main plotting function of GEMDAT. it takes two mandatory arguments:
    - data: which is a Data object that can create/format/read/write many
    types of data
    - plots, a list of plot names, or just a plot name for the plot you want.
    - Optional arguments which are passed down to the plotting functions.

    Parameters
    ----------
    data : Data
        data
    plots : Union[List[str],str]
        plots
    kwargs :
        kwargs

    Returns
    -------
    None

    """

    # Convert plots to list, if it is not already a list
    if not isinstance(plots, list):
        plots = [plots]

    for plot in plots:
        plot_function = getattr(available_plots, plot)
        plot_function(data, **kwargs)


def plot_all(data: Data, **kwargs) -> None:
    """The Plot All function finds out which plots are available for plotting,
    and plots those.

    Parameters
    ----------
    data : Data
        data
    kwargs :
        kwargs

    Returns
    -------
    None
    """
    plot(data=data, plots=available_plots.__all__, **kwargs)
