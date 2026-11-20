from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._axes import get_axes

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from gemdat.metrics import ArrheniusFit


def arrhenius(
    *,
    fit: ArrheniusFit,
    show_std: bool = True,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure | matplotlib.axes.Axes:
    """Plot Arrhenius fit.

    Parameters
    ----------
    fit
        ArrheniusFit instance.
    show_std
        If True, show error bars (from diffusivities_std) and a ±1σ fit band
        (from cov).
    ax : matplotlib.axes.Axes | None
        Draw into this axes instead of a new figure; the axes is then returned.

    Returns
    -------
    fig : matplotlib.figure.Figure | matplotlib.axes.Axes
        Output figure, or `ax` if given.
    """
    fig, ax = get_axes(ax)

    T = fit.temperatures
    x = 1000.0 / T
    y = np.log(fit.diffusivities)

    if show_std and getattr(fit, 'diffusivities_std', None) is not None:
        yerr = fit.diffusivities_std / fit.diffusivities  # sigma_lnD ≈ sigma_D/D
        ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=3, label='data')
    else:
        ax.scatter(x, y, label='data')

    last_color = ax.lines[-1].get_color()

    # Fit line
    t_line = np.linspace(float(T.min()), float(T.max()), 200)
    x_line = 1000.0 / t_line
    ln_line = fit.intercept + fit.slope * (1.0 / t_line)
    ax.plot(x_line, ln_line, label='fit', color=last_color)

    # ±1σ band from covariance (in ln-space)
    if show_std and getattr(fit, 'cov', None) is not None:
        v = np.column_stack([1.0 / t_line, np.ones_like(t_line)])
        var = np.einsum('ij,jk,ik->i', v, fit.cov, v)
        std = np.sqrt(np.maximum(var, 0.0))
        ax.fill_between(
            x_line, ln_line - std, ln_line + std, alpha=0.2, label='±1σ', color=last_color
        )

    ax.set_xlabel(r'1000/T (K$^{-1}$)')
    ax.set_ylabel(r'ln(D)')

    ax.legend()
    return ax if fig is None else fig
