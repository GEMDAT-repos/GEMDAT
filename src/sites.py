import numpy as np
from pymatgen.core import Lattice


def calculate_atom_sites(*, coords, site_coords, lattice: Lattice,
                         dist_close: float):
    """Calculate nearest site for each atom coordinate.

    Note: This is a slow operation, because a pairwise distance matrix between all `coords` and
    all `site_coords` has to be generated. This includes lattice translations. The nearest site
    may be in the neighbouring unit cell.

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
    atom_sites : np.ndarray
        Output array with site locations for each atom at each time step [time, atom].
        The value corresponds to the index in the `site_coords`.
        -1 indicates that atom is not at any site.
    """
    atom_sites = []

    for atom_index, atom_coords in enumerate(coords.swapaxes(0, 1)):
        pdist = lattice.get_all_distances(atom_coords, site_coords)

        # index of nearest site
        nearest = pdist.argmin(axis=1, keepdims=True)

        # True if atom is close enough to a site
        is_at_site = np.take_along_axis(pdist, nearest, axis=1) < dist_close

        # Site index when close, -1 when in transition
        atom_site = np.where(is_at_site, nearest, -1)

        atom_sites.append(atom_site)

    return np.hstack(atom_sites)


def calculate_transitions(*, atom_sites: np.ndarray) -> np.ndarray:
    """Find transitions between sites.

    Parameters
    ----------
    atom_sites : np.ndarray
        Input array with atom sites

    Returns
    -------
    all_transitions : np.ndarray
        Output array with transition events.
        Contains 5 columns: atom index, time start, time stop, site start, site stop
    """

    all_transitions = []

    for atom_index, atom_site in enumerate(atom_sites.T):

        # Indices when atom jumps to new or back to same site
        i, = np.nonzero((atom_site != np.roll(atom_site, shift=1))
                        & (atom_site >= 0))

        # Log transition events
        i_event = np.nonzero(atom_site[i] != np.roll(atom_site[i], shift=-1))
        time_start = i[i_event]
        time_stop = np.roll(i, shift=-1)[i_event]
        transitions = np.vstack([
            np.ones_like(time_start) * atom_index,
            atom_site[time_start],
            atom_site[time_stop],
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


def split_transitions_in_parts(all_transitions: np.ndarray,
                               n_steps: int,
                               n_parts=10) -> list[np.ndarray]:
    """Split list of transition events into equal parts in time.

    Parameters
    ----------
    all_transitions : np.ndarray
        Input array with transition events
    n_steps : int
        Number of time steps
    n_parts : int, optional
        Number of parts to split into

    Returns
    -------
    transitions_parts : np.ndarray
        Sorted list of transition events split into equal parts.
        The first dimension corresponds to `n_parts`.
    """
    col = 4

    bins = np.linspace(0, n_steps + 1, n_parts + 1, dtype=int)
    parts = np.digitize(all_transitions[:, col], bins=bins)
    parts = parts[parts.argsort()]
    splits = np.unique(parts, return_index=True)[1][1:]

    sorted_transitions = all_transitions[all_transitions[:, col].argsort()]

    return np.split(sorted_transitions, splits)


def calculate_occupancy(atom_sites: np.ndarray) -> dict[int, int]:
    """Calculate occupancy per site.

    Parameters
    ----------
    atom_sites : np.ndarray
        Input array with atom sites

    Returns
    -------
    dict[int, int]
        For each site, count for how many time steps it is occupied by an atom
    """
    unq, counts = np.unique(atom_sites, return_counts=True)
    return dict(zip(unq, counts))


if __name__ == '__main__':
    from gemdat import SimulationData

    vasp_xml = '/run/media/stef/Scratch/md-analysis-matlab-example/vasprun.xml'

    equilibration_steps = 1250
    diffusing_element = 'Li'

    data = SimulationData.from_vasprun(vasp_xml, cache='vasprun.xml.cache')

    data.calculate_all(
        equilibration_steps=equilibration_steps,
        diffusing_element='Li',
    )

    dist_close = 2 * data.extras['vibration_amplitude']

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

    n_parts = 10

    traj_coords = data.trajectory_coords

    species = data.species
    diffusing_idx = np.argwhere([e.name == diffusing_element
                                 for e in species]).flatten()

    diff_coords = traj_coords[equilibration_steps:, diffusing_idx, :]

    assert diff_coords.shape == (73750, 48, 3)

    atom_sites = calculate_atom_sites(coords=diff_coords,
                                      site_coords=site_coords,
                                      lattice=lattice,
                                      dist_close=dist_close)

    assert atom_sites.shape == (73750, 48)
    assert atom_sites.sum() == 9228360

    all_transitions = calculate_transitions(atom_sites=atom_sites)

    assert all_transitions.shape == (1336, 5)

    n_diffusing = len(diffusing_idx)

    transitions = calculate_transitions_matrix(all_transitions,
                                               n_diffusing=n_diffusing)

    assert transitions.shape == (48, 48)

    n_steps = len(diff_coords)
    split_transitions = split_transitions_in_parts(all_transitions, n_steps,
                                                   n_parts)
    success = np.stack([
        calculate_transitions_matrix(part, n_diffusing=n_diffusing)
        for part in split_transitions
    ])

    assert len(split_transitions) == n_parts
    assert success.shape == (n_parts, 48, 48)
    assert np.sum(success[0]) == 134
    assert np.sum(success[9]) == 142

    occupancy = calculate_occupancy(atom_sites)

    assert occupancy[-1] == 3015185
    assert occupancy[0] == 1706
    assert occupancy[43] == 6350

    split_atom_sites = np.split(atom_sites, n_parts)
    occupancy_parts = [calculate_occupancy(part) for part in split_atom_sites]

    assert len(occupancy_parts) == n_parts

    assert occupancy_parts[0][0] == 241
    assert occupancy_parts[0][43] == 1231
    assert occupancy_parts[9][0] == 87
    assert occupancy_parts[9][43] == 391
