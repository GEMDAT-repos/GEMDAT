from functools import cached_property

import numpy as np

from pymatgen.io import vasp


class Data():

    equilibration_steps = 1250
    vasprun = None

    @classmethod
    def from_vasprun(cls, run: vasp.outputs.Vasprun):
        data = cls()
        data.vasprun = run
        return data
        # probably need to do some correct initialization here maybe pydantic

    @cached_property
    def trajectory(self):
        return self.vasprun.get_trajectory()

    @cached_property
    def cell_offsets(self) -> np.ndarray:
        """Calculate cell offsets from trajectory."""
        assert not self.trajectory.coords_are_displacement
        coords = self.trajectory.coords
        return self.cell_offsets_from_coords(coords)

    def cell_offsets_from_coords(self, coords: np.ndarray) -> np.ndarray:
        """Calculate cell offsets from starting position.

        For example, if a site is at [0, 0, 0.9] -> [0, 0, 0.1]
        assume it has jumped to the next cell: [0, 0, 1.1]

        Parameters
        ----------
        coords : np.ndarray[i, j, k]
            3-dimensional numpy array with dimensions i: time_steps, j: sites, k: coordinates

        Returns
        -------
        offsets : np.ndarray[i, j, k]
            Integer array with unit cell offset vectors.
        """
        first = coords[0, np.newaxis]
        diff = np.diff(coords, axis=0, prepend=first)

        digits = np.digitize(diff, bins=[0.5, -0.5]) - 1

        offsets = np.cumsum(digits, axis=0)
        return offsets

    def lengths(self, vectors: np.ndarray,
                metric_tensor: np.ndarray) -> np.ndarray:
        """Calculate vector lengths using the metric tensor (Dunitz 1078,
        p227).

        Parameters
        ----------
        vectors : np.ndarray[i, j, k]
            Vectors as in fractional coordinates
        metric_tensor : np.ndarray
            Metric tensor for the lattice

        Returns
        -------
        lengths : np.ndarray
            Vector lengths
        """
        tmp = np.dot(vectors, metric_tensor)
        total_displacement = np.einsum('ij,ji->i', tmp, vectors.T)
        assert total_displacement.shape[0] == vectors.shape[0]
        assert total_displacement.ndim == 1
        return np.sqrt(total_displacement)

    @cached_property
    def displacements(self) -> np.ndarray:
        """Calculate displacements from first set of positions.

        Corrects for elements jumping to the next unit cell.

        Parameters
        ----------
        traj_coords : np.array[i, j, k]
            3-dimensional numpy array with dimensions i: time_steps, j: sites, k: coordinates

        Returns
        -------
        displacements : np.ndarray[i, j]
            Displacements from first set of positions.
        """
        offsets = self.cell_offsets_from_coords(self.trajectory.coords)

        corrected_coords = self.trajectory.coords + offsets

        displacements = []

        first = corrected_coords[self.equilibration_steps]

        for disp in corrected_coords[self.equilibration_steps:]:
            diff_vectors = disp - first
            lengths = self.lengths(diff_vectors,
                                   metric_tensor=self.lattice.metric_tensor)
            displacements.append(lengths)

        displacements = np.array(displacements)

        return displacements.T

    @cached_property
    def structure(self):
        return self.vasprun.structures[0]

    @cached_property
    def species(self):
        return self.structure.species

    @cached_property
    def lattice(self):
        return self.structure.lattice
