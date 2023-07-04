from dataclasses import asdict
from typing import List, Optional, Union

import gemdat.plots as available_plots

from .data import SimulationData


def plot(plots: Union[List[str], str],
         data: Optional[SimulationData] = None,
         **kwargs) -> None:
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
    None
    """

    # Convert plots to list, if it is not already a list
    if not isinstance(plots, list):
        plots = [plots]

    # extract data if present, but prioritise kwargs
    if data:
        kwargs = {**asdict(data), **kwargs}

    for plot in plots:
        plot_function = getattr(available_plots, plot)
        plot_function(**kwargs)


def plot_all(**kwargs) -> None:
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
    None
    """
    plot(plots=available_plots.__all__, **kwargs)
