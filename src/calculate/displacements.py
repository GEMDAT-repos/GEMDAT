import numpy as np


class Displacements:

    @staticmethod
    def calculate_all(data, extras) -> dict:
        """Calculate displacement properties.

        Parameters
        ----------
        data : SimulationData
            Input simulation data
        extras : SimpleNamespace
            Extra variables

        Returns
        -------
        extras : dict[str, float]
            Dictionary with calculated parameters
        """
        cell_offsets = Displacements.cell_offsets(data.trajectory_coords),

        displacements = Displacements.displacements(data.trajectory_coords,
                                                    data.lattice,
                                                    extras.equilibration_steps)

        diff_displacements = Displacements.diff_displacements(
            displacements=displacements,
            diffusing_element=extras.diffusing_element,
            species=data.species)

        return {
            'cell_offsets': cell_offsets,
            'displacements': displacements,
            'diff_displacements': diff_displacements,
        }

    @staticmethod
    def cell_offsets_from_coords(coords: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def cell_offsets(trajectory_coords) -> np.ndarray:
        """Calculate cell offsets from trajectory."""
        coords = trajectory_coords
        return Displacements.cell_offsets_from_coords(coords)

    @staticmethod
    def lengths(vectors: np.ndarray, metric_tensor: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def displacements(trajectory_coords,
                      lattice,
                      equilibration_steps=1250) -> np.ndarray:
        """Calculate displacements from first set of positions.

        Corrects for elements jumping to the next unit cell.

        Parameters
        ----------
        trajectory_coords : np.array[i, j, k]
            3-dimensional numpy array with dimensions i: time_steps, j: sites, k: coordinates
        lattice:


        Returns
        -------
        displacements : np.ndarray[i, j]
            Displacementss from first set of positions.
        """
        offsets = Displacements.cell_offsets_from_coords(trajectory_coords)

        corrected_coords = trajectory_coords + offsets

        displacements = []

        first = corrected_coords[equilibration_steps]

        for disp in corrected_coords[equilibration_steps:]:
            diff_vectors = disp - first
            lengths = Displacements.lengths(
                diff_vectors, metric_tensor=lattice.metric_tensor)
            displacements.append(lengths)

        displacements = np.array(displacements)

        return displacements.T

    @staticmethod
    def diff_displacements(*, diffusing_element='Li', displacements, species):
        idx = np.argwhere([e.name == diffusing_element for e in species])
        return displacements[idx].squeeze()
