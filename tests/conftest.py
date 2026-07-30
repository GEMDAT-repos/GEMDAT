from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Species

from gemdat.orientations import Orientations
from gemdat.trajectory import Trajectory


@pytest.fixture()
def trajectory():
    coords = np.array(
        [
            [[0.2, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.5], [0.0, 0.0, 0.5]],
            [[0.4, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.5], [0.0, 0.0, 0.5]],
            [[0.6, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.5], [0.0, 0.0, 0.5]],
            [[0.8, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.5], [0.0, 0.0, 0.5]],
            [[0.1, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.5], [0.0, 0.0, 0.5]],
        ]
    )

    return Trajectory(
        species=[Species('B'), Species('Si'), Species('S'), Species('C')],
        coords=coords,
        lattice=np.eye(3),
        metadata={'temperature': 123},
        time_step=1,
    )


@pytest.fixture()
def variable_lattice_trajectory():
    """Trajectory with a per-frame (non-constant) lattice, as in an NPT run."""
    coords = np.array(
        [
            [[0.2, 0.0, 0.0], [0.0, 0.0, 0.5]],
            [[0.4, 0.0, 0.0], [0.0, 0.0, 0.5]],
            [[0.6, 0.0, 0.0], [0.0, 0.0, 0.5]],
        ]
    )
    lattices = np.array([np.eye(3) * s for s in (1.0, 1.1, 1.2)])

    return Trajectory(
        species=[Species('Li'), Species('S')],
        coords=coords,
        lattice=lattices,
        constant_lattice=False,
        metadata={'temperature': 123},
        time_step=1,
    )


@pytest.fixture()
def orientations(trajectory):
    center_type = 'B'
    satellite_type = 'Si'
    return Orientations(trajectory, center_type, satellite_type)


@pytest.fixture()
def trajectory_list():
    rng = np.random.default_rng(0)

    n_frames = 200
    lattice = np.eye(3)
    species = [Species('B'), Species('Si'), Species('S'), Species('C')]

    static_sites = np.array(
        [[0.0, 0.0, 0.5], [0.0, 0.0, 0.5], [0.0, 0.0, 0.5]],
        dtype=float,
    )

    out = []
    for temperature, step in [
        (400.0, 1.0e-3),
        (500.0, 1.2e-3),
        (600.0, 1.4e-3),
        (700.0, 1.6e-3),
    ]:
        coords = np.empty((n_frames, 4, 3), dtype=float)

        # diffusing B atom: small random walk (stays near 0.2, won’t wrap)
        disp = rng.normal(scale=step, size=(n_frames, 3)).cumsum(axis=0)
        coords[:, 0, :] = 0.2 + disp

        # other atoms fixed
        coords[:, 1:, :] = static_sites[None, :, :]

        out.append(
            Trajectory(
                species=species,
                coords=coords,
                lattice=lattice,
                metadata={'temperature': temperature},
                time_step=1,
            )
        )

    return out
