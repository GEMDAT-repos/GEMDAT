from __future__ import annotations

from types import ModuleType

from gemdat import plots as plots_default
from gemdat.plots import matplotlib as plots_matplotlib
from gemdat.plots import plotly as plots_plotly


def plot_backend(func):
    """Decorator to switch plotting backend."""

    def wrap(*args, backend: str | None = None, **kwargs):
        module: ModuleType

        if backend is None:
            module = plots_default
        elif backend in ('mpl', 'matplotlib'):
            module = plots_matplotlib
        elif backend == 'plotly':
            module = plots_plotly
        else:
            raise ValueError(f'No such backend: {backend}')

        result = func(*args, module=module, **kwargs)

        return result

    wrap.__doc__ = func.__doc__
    wrap.__doc__ += """

Parameters
---------
backend : str
    Choose plotting backend. Options: matplotlib, mpl, plotly
    Defaults to plotly unless the plot is only available in matplotlib.

Returns
-------
fig : plotly.graph_objects.Figure or matplotlib.figure.Figure depending on backend.
    Output figure
"""

    return wrap
