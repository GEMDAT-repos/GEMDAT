from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import plotly.graph_objects as go
    from pymatgen.core import Lattice, Structure

    from gemdat.volume import Volume


def density(
    volume: Volume,
    *,
    structure: Structure | None = None,
    force_lattice: Lattice | None = None,
) -> go.Figure:
    """Create density plot from volume and structure.

    Uses plotly as plotting backend.

    Arguments
    ---------
    volume : Volume
        Input volume
    structure : Structure, optional
        Input structure
    force_lattice : Lattice | None
        Plot volume and structure using this lattice as a basis.
        Overrides the default, which is to use `volume.lattice`
        and `structure.lattice` where applicable.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Output as plotly figure
    """
    from ._plot3d import plot_3d

    return plot_3d(volume=volume, structure=structure, lattice=force_lattice)
