from __future__ import annotations

from pathlib import Path

import numpy as np
import scipp as sc
from numpy.testing import assert_allclose
from pymatgen.core import Lattice, Species

from gemdat.trajectory import Trajectory


def test_trajectory(trajectory):
    assert isinstance(trajectory, Trajectory)
    assert trajectory.species == [
        Species('B'),
        Species('Si'),
        Species('S'),
        Species('C'),
    ]
    assert trajectory.positions.shape == (5, 4, 3)
    assert trajectory.metadata == {'temperature': 123}


def test_slice(trajectory):
    sliced = trajectory[2:]

    assert isinstance(sliced, Trajectory)
    assert sliced.species == trajectory.species
    assert sliced.positions.shape == (3, 4, 3)
    assert sliced.metadata == trajectory.metadata


def test_filter(trajectory):
    t = trajectory.filter('C')
    assert t.species == [Species('C')]
    assert np.all(t.positions == [0.0, 0.0, 0.5])


def test_get_lattice(trajectory):
    lattice = trajectory.get_lattice()
    expected_lattice = Lattice(np.eye(3))

    assert isinstance(lattice, Lattice)
    assert lattice == expected_lattice


def test_caching(trajectory, tmpdir):
    cachefile = tmpdir / 'trajectory.cache'
    trajectory.to_cache(cachefile)

    assert cachefile.exists()

    t2 = Trajectory.from_cache(cachefile)

    assert trajectory.species == t2.species
    assert trajectory.metadata == t2.metadata
    assert trajectory.time_step == t2.time_step

    assert_allclose(trajectory.lattice, t2.lattice)
    assert_allclose(trajectory.base_positions, t2.base_positions)
    assert_allclose(trajectory.positions, t2.positions)


def test_displacements_property(trajectory):
    trajectory = trajectory.filter(['B', 'C'])
    trajectory.to_positions()

    assert_allclose(
        trajectory.displacements,
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.3, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
    )

    assert trajectory.coords_are_displacement


def test_positions_property(trajectory):
    trajectory.to_displacements()

    assert_allclose(
        trajectory.filter(['B', 'S']).positions,
        [
            [[0.2, 0.0, 0.0], [0.0, 0.0, 0.5]],
            [[0.4, 0.0, 0.0], [0.0, 0.0, 0.5]],
            [[0.6, 0.0, 0.0], [0.0, 0.0, 0.5]],
            [[0.8, 0.0, 0.0], [0.0, 0.0, 0.5]],
            [[0.1, 0.0, 0.0], [0.0, 0.0, 0.5]],
        ],
    )

    assert not trajectory.coords_are_displacement


def test_drift_correction(trajectory):
    drift = trajectory.drift(fixed_species='B')
    assert drift.shape == (5, 1, 3)
    global_drift = np.mean(drift, axis=0)
    assert_allclose(global_drift, [[0.18, 0.0, 0.0]])

    t2 = trajectory.apply_drift_correction(fixed_species='B')
    global_drift2 = np.mean(t2.drift(fixed_species='B'), axis=0)

    # drift must now be effectively removed
    assert_allclose(global_drift2, [[0.0, 0.0, 0.0]])


def test_distances_from_base_position(trajectory):
    distances = trajectory.filter(['B', 'Si']).distances_from_base_position()
    assert_allclose(
        distances,
        [
            [0.0, 0.2, 0.4, 0.6, 0.9],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
    )


def test_cumulative_displacements(trajectory):
    displacements = trajectory.filter(['B', 'C']).cumulative_displacements
    assert_allclose(
        displacements,
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.4, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.6, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.9, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
    )


def test_filter_similar_names(trajectory):
    subtrajectory = trajectory.filter('Si')
    assert subtrajectory.species == [Species('Si')]
    subtrajectory = trajectory.filter(['S', 'Si'])
    assert subtrajectory.species == [Species('Si'), Species('S')]
    subtrajectory = trajectory.filter('S')
    assert subtrajectory.species == [Species('S')]


def test_trajectory_extend(trajectory):
    trajectory.extend(trajectory)

    assert isinstance(trajectory, Trajectory)
    assert len(trajectory) == 10
    assert_allclose(
        trajectory.positions[:, 0, 0],
        [0.2, 0.4, 0.6, 0.8, 0.1, 0.2, 0.4, 0.6, 0.8, 0.1],
    )


def test_mean_squared_displacement(trajectory):
    msd = trajectory.mean_squared_displacement()
    assert len(msd) == 4
    assert_allclose(msd[0], [0.0, 0.0525, 0.19, 0.425, 0.81])
    assert isinstance(msd, np.ndarray)
    assert_allclose(msd.mean(), 0.073875)


def test_kinisi_cache(trajectory):
    diff = trajectory.to_kinisi_diffusion_analyzer(specie='B', progress=False)
    assert trajectory.kinisi_diffusion_analyzer_cache is not None

    diff2 = trajectory.kinisi_diffusion_analyzer_cache

    assert sc.identical(diff.da, diff2.da)

    assert_allclose(diff.msd.values, diff2.msd.values)
    assert_allclose(diff.msd.variances, diff2.msd.variances)
    assert_allclose(diff.dt.values, diff2.dt.values)


def test_kinisi_mean_squared_displacement(trajectory):
    diff = trajectory.to_kinisi_diffusion_analyzer(specie='B', progress=False)
    assert diff.n_atoms == 1
    msd = diff.msd
    assert len(msd) == 4
    assert_allclose(msd.values, [0.042, 0.1525, 0.33666667, 0.585])
    assert isinstance(msd, sc._scipp.core.Variable)
    assert msd.unit == 'Å^2'
    assert_allclose(msd.variances, [0.000204, 0.00297, 0.01658, 0.081])
    dt = diff.dt
    assert isinstance(dt, sc._scipp.core.Variable)
    assert dt.unit == 'ps'
    assert_allclose(dt.values, [1.e+12, 2.e+12, 3.e+12, 4.e+12])
    rng = np.random.RandomState(42)
    diff.diffusion(start_dt=sc.scalar(0, unit='ps'), random_state=rng, progress=False)
    assert diff.D.values.mean() == 2.046301680845264e-18


def test_from_lammps():
    data_dir = Path(__file__).parent / 'data' / 'lammps'

    traj = Trajectory.from_lammps(
        coords_file=data_dir / 'lammps_coords.xyz',
        data_file=data_dir / 'lammps_data.txt',
        temperature=700,
        time_step=2,
    )

    assert traj.positions.shape == (4, 80, 3)
    assert len(traj.species) == 80
    assert traj.time_step_ps == 2


def test_from_gromacs():
    data_dir = Path(__file__).parent / 'data' / 'gromacs'

    traj = Trajectory.from_gromacs(
        topology_file=data_dir / 'gromacs_topology.tpr',
        coords_file=data_dir / 'gromacs_short_trajectory.xtc',
        temperature=300,
    )

    assert traj.positions.shape == (251, 18943, 3)
    assert len(traj.species) == 18943
    assert traj.time_step_ps == 2
