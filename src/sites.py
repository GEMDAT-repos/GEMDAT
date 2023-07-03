import numpy as np
from pymatgen.core import Lattice


def calculate_transitions(*, coords, site_coords, lattice: Lattice,
                          dist_close: float):
    """Find transitions between sites.

    Parameters
    ----------
    coords : np.ndarray
        Input array with atom coordinates [time, atom, (x, y, z)]
    site_coords : np.ndarray
        Input array with site coordinates [site, (x, y, z)]
    lattice : Lattice
        Unit cell parameters
    dist_close : float
        Atoms within this distance (in Angstrom) are considered to be close to a site

    Returns
    -------
    all_transitions : np.ndarray
        Output array with transition events.
        Contains 5 columns: atom index, time start, time stop, site start, site stop
    """

    all_transitions = []

    for atom_index, atom_coords in enumerate(coords.swapaxes(0, 1)):
        pdist = lattice.get_all_distances(atom_coords, site_coords)

        # index of nearest site
        nearest = pdist.argmin(axis=1, keepdims=True)

        # True if atom is close enough to a site
        is_at_site = np.take_along_axis(pdist, nearest, axis=1) < dist_close

        # Site index when close, -1 when in transition
        atom_site = np.where(is_at_site, nearest, -1)

        # Indices when atom jumps to new or back to same site
        i, _ = np.nonzero((atom_site != np.roll(atom_site, shift=1))
                          & (atom_site >= 0))

        # Log transition events
        i_event = np.nonzero(atom_site[i,
                                       0] != np.roll(atom_site[i,
                                                               0], shift=-1))
        time_start = i[i_event]
        time_stop = np.roll(i, shift=-1)[i_event]
        transitions = np.vstack([
            np.ones_like(time_start) * atom_index,
            atom_site[time_start, 0],
            atom_site[time_stop, 0],
            time_start,
            time_stop,
        ]).T

        # Drop last event (side effect of np.roll)
        transitions = transitions[:-1]
        all_transitions.append(transitions)

    return np.vstack(all_transitions)


def calculate_transitions_matrix(all_transitions: np.ndarray,
                                 n_diffusing: int) -> np.ndarray:
    """Convert list of transition events to dense transitions matrix.

    Parameters
    ----------
    all_transitions : np.ndarray
        Input array with transition events
    n_diffusing : int
        Number of diffusing elements. This defines the shape of the output matrix.

    Returns
    -------
    np.ndarray
        Square matrix with number of each transitions
    """
    start_col = 1  # transition starts
    stop_col = 2  # transition stop

    transitions = np.zeros((n_diffusing, n_diffusing), dtype=int)
    idx, counts = np.unique(all_transitions[:, [start_col, stop_col]],
                            return_counts=True,
                            axis=0)
    start_idx, stop_idx = idx.T
    transitions[start_idx, stop_idx] = counts
    return transitions


if __name__ == '__main__':
    from gemdat import Data

    vasp_xml = '/run/media/stef/Scratch/md-analysis-matlab-example/vasprun.xml'

    data = Data.from_vasprun(vasp_xml, cache='vasprun.xml.cache')

    dist_close = 2 * 0.4839  # 'vibration_amplitude' in simdata

    lattice = data.lattice

    from gemdat.io import load_known_material

    structure = load_known_material('argyrodite')

    site_coords = structure.frac_coords

    pdist = lattice.get_all_distances(site_coords, site_coords)
    min_dist = np.min(pdist[np.triu_indices_from(pdist, k=1)])

    if min_dist < 2 * dist_close:
        # Crystallographic sites are overlapping with the chosen dist_close, making it smaller
        dist_close = (0.5 * min_dist) - 0.005

        # Two crystallographic sites are within half an Angstrom of each other
        # This is NOT realistic, check/change the given crystallographic site
        if dist_close * 2 < 0.5:
            idx = np.argwhere(pdist == min_dist)
            collisions = [
                f'{structure.sites[i]}-{structure.sites[j]}' for i, j in idx
            ]

            lines = []

            for i, j in idx:
                site_i = structure.sites[i]
                site_j = structure.sites[j]
                lines.append('\nToo close:')
                lines.append(
                    '{site_i.specie.name}({i}) {site_i.frac_coords} - ')
                lines.append(f'{site_j.specie.name}({j}) {site_j.frac_coords}')

            msg = ''.join(lines)

            raise ValueError(
                f'Crystallographic sites are too close together (expected: >{dist_close*2:.4f}, '
                f'got: {min_dist:.4f} for {msg}')

    equilibration_steps = 1250

    diffusing_element = 'Li'
    traj_coords = data.trajectory_coords

    species = data.species
    diffusing_idx = np.argwhere([e.name == diffusing_element
                                 for e in species]).flatten()

    diff_coords = traj_coords[equilibration_steps:, diffusing_idx, :]

    assert diff_coords.shape == (73750, 48, 3)

    all_transitions = calculate_transitions(coords=diff_coords,
                                            lattice=lattice,
                                            site_coords=site_coords,
                                            dist_close=dist_close)

    assert all_transitions.shape == (1336, 5)

    n_diffusing = len(diffusing_idx)

    transitions = calculate_transitions_matrix(all_transitions,
                                               n_diffusing=n_diffusing)

    assert transitions.shape == (48, 48)
