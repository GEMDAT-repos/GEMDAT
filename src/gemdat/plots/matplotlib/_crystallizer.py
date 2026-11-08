from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from pymatgen.electronic_structure import plotter

from gemdat.plots._shared import (
    _crystallizer_site_images,
    _crystallizer_sites,
    density_matching_cmap_mpl,
)

if TYPE_CHECKING:
    import matplotlib.figure

    from gemdat.crystallizer import CrystallizerScan


def crystallizer_sites(
    *,
    scan: CrystallizerScan,
    symprec: float | None = None,
    specie: str | None = None,
    background_level: float = 0.1,
) -> matplotlib.figure.Figure:
    """Plot the idealised sites of one species of a symmetry fit in 3D, inside
    the density of their species.

    The density is drawn as one grey point per voxel above `background_level`,
    darker where denser. Sites are coloured by orbit (symmetry-equivalent set)
    and labelled with their Wyckoff letter and multiplicity.

    Parameters
    ----------
    scan : CrystallizerScan
        Scan from [gemdat.crystallizer.Crystallizer.scan][].
    symprec : float | None
        Tolerance (Ångstrom) of the fit to plot, e.g. `result.symprec` or
        `level.symprec`. Defaults to the best (highest) space group of the scan.
    specie : str | None
        Species whose sites and density to plot, defaults to the mobile species.
        A framework species' sites are its time-averaged atom positions.
    background_level : float
        Voxels below this fraction of the maximum density are not drawn.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    title, sites, density = _crystallizer_sites(scan, symprec, specie)
    lattice = sites.lattice

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    plotter.plot_lattice_vectors(lattice, ax=ax, linewidth=1)

    data = density.data / density.data.max()
    voxels = np.argwhere(data > background_level)
    values = data[tuple(voxels.T)]
    xyz = lattice.get_cartesian_coords((voxels + 0.5) / density.dims)
    colors = np.zeros((len(values), 4))
    colors[:, 3] = 0.6 * values  # black, more opaque where denser
    ax.scatter(*xyz.T, s=15, color=colors, linewidths=0)

    labels = np.array(sites.labels)
    for n, label in enumerate(dict.fromkeys(sites.labels)):
        coords = sites.cart_coords[labels == label]
        ax.scatter(
            *coords.T,
            s=20,
            color=f'C{n % 10}',
            edgecolors='black',
            linewidths=0.5,
            depthshade=False,
            label=label,
        )

    ax.set(title=title, xlabel='x (Å)', ylabel='y (Å)', zlabel='z (Å)')
    ax.set_aspect('equal')
    ax.legend(loc='upper left', title='orbit (sites)')

    return fig


def crystallizer_site_images(
    *,
    scan: CrystallizerScan,
    symprec: float | None = None,
    specie: str | None = None,
    site: int | None = None,
    width: float = 1.5,
    resolution: float = 0.05,
) -> matplotlib.figure.Figure:
    """Plot one ideal site, the images the symmetry operations map onto it, and
    the density around it.

    The images are the density-derived sites of this site's orbit, each mapped
    onto it by one symmetry operation (the identity gives the site itself). The
    ideal position is (up to a common origin shift) their mean, see
    [gemdat.symmetry.SymmetryAnalyzer.idealise][], so their scatter shows
    how well this site obeys the symmetry. Three Cartesian views (xy, xz, yz)
    are centred on the ideal site, each over the density averaged along the
    viewing direction within the plotted box.

    Parameters
    ----------
    scan : CrystallizerScan
        Scan from [gemdat.crystallizer.Crystallizer.scan][].
    symprec : float | None
        Tolerance (Ångstrom) of the fit to plot. Defaults to the best (highest)
        space group of the scan.
    specie : str | None
        Species of the site to plot when `site` is not given, defaults to the
        mobile species.
    site : int | None
        Index of the site in `scan.geometry`. Defaults to the first site of
        `specie`. The background is the density of the site's species.
    width : float
        Half-width (Ångstrom) of the plotted box around the ideal site.
    resolution : float
        Spacing (Ångstrom) at which the density is sampled for the background.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    title, ticks, box, images = _crystallizer_site_images(
        scan, symprec, specie, site, width, resolution
    )

    cmap = density_matching_cmap_mpl()
    views = (2, 1, 0)
    projections = {axis: box.mean(axis=axis).T for axis in views}
    vmax = max(projection.max() for projection in projections.values())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
    for axis, ax in zip(views, axes):
        x, y = (i for i in range(3) if i != axis)
        mesh = ax.pcolormesh(
            ticks, ticks, projections[axis], cmap=cmap, vmin=0, vmax=vmax, shading='nearest'
        )
        ax.scatter(
            images[:, x],
            images[:, y],
            s=15,
            color='tab:blue',
            edgecolors='white',
            linewidths=0.5,
            label='images',
            zorder=3,  # on top of the star, which hides a tight cluster
        )
        ax.scatter(0, 0, s=150, marker='*', color='tab:red', label='ideal')
        ax.set_xlabel(f'{"xyz"[x]} (Å)')
        ax.set_ylabel(f'{"xyz"[y]} (Å)')
        ax.set_title(f'{"xyz"[x]}{"xyz"[y]} view')
        ax.set_aspect('equal')

    fig.colorbar(mesh, ax=axes, label='counts / voxel', shrink=0.8)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside upper right')
    fig.suptitle(title)

    return fig
