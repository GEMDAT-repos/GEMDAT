from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from types import SimpleNamespace

    from gemdat.trajectory import Trajectory


class Displacements:

    @staticmethod
    def calculate_all(trajectory: Trajectory, extras: SimpleNamespace) -> dict:
        """Calculate displacement properties.

        Parameters
        ----------
        trajectory : Trajectory
            Input simulation data
        extras : SimpleNamespace
            Extra variables

        Returns
        -------
        extras : dict[str, float]
            Dictionary with calculated parameters
        """
        Displacements.cell_offsets(trajectory)

        displacements = Displacements.displacements(trajectory)

        Displacements.diff_displacements(
            displacements=displacements,
            diffusing_element=extras.diffusing_element,
            species=trajectory.species)
