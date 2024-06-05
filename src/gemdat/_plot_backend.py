from __future__ import annotations


def plot_backend(func):
    """Decorator to switch plotting backend."""

    def wrap(*args, backend: str | None = None, **kwargs):
        if backend is None:
            from gemdat import plots as _backend
        elif backend in ('mpl', 'matplotlib'):
            from gemdat.plots import matplotlib as _backend
        elif backend == 'plotly':
            from gemdat.plots import plotly as _backend
        else:
            raise ValueError(f'No such backend: {backend}')

        result = func(*args, _backend=_backend, **kwargs)

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
