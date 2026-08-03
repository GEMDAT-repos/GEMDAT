from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipp as sc
from numpy.testing import assert_allclose
from pymatgen.core import Element, Lattice, Species

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


def test_get_lattice_variable(variable_lattice_trajectory):
    # See GitHub issue #394: get_lattice() without an index used to crash for
    # a non-constant lattice.
    traj = variable_lattice_trajectory

    # With an index, return that frame's lattice.
    assert traj.get_lattice(0) == Lattice(np.eye(3))
    assert traj.get_lattice(1) == Lattice(np.eye(3) * 1.1)
    assert traj.get_lattice(2) == Lattice(np.eye(3) * 1.2)

    # Without an index, a non-constant lattice is ambiguous -> raise.
    with pytest.raises(ValueError, match='not constant'):
        traj.get_lattice()


def test_filter_variable_lattice(variable_lattice_trajectory):
    # issue #394: filter() must preserve the per-frame lattice.
    traj = variable_lattice_trajectory
    filtered = traj.filter('Li')

    assert filtered.species == [Species('Li')]
    assert filtered.constant_lattice is False
    assert_allclose(filtered.lattice, traj.lattice)


def test_mean_squared_displacement_variable_lattice(variable_lattice_trajectory):
    # issue #394: mean_squared_displacement() called get_lattice() with no index.
    msd = variable_lattice_trajectory.mean_squared_displacement()

    assert msd.shape == (2, 3)


LAMMPS_DATA_FILE = """LAMMPS data file for tests

2 atoms
2 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Masses

1 6.941
2 32.06

Atoms # atomic

1 1 1.0 0.0 0.0
2 2 5.0 5.0 5.0
"""

# Cubic box doubling every frame. Atom 1 sits at fractional x = 0.1, 0.2, 0.3,
# atom 2 stays put at the box centre. Cartesian coordinates, as in a real dump.
LAMMPS_NPT_DUMP_FILE = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z
1 1 1.0 0.0 0.0
2 2 5.0 5.0 5.0
ITEM: TIMESTEP
1
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 20.0
0.0 20.0
0.0 20.0
ITEM: ATOMS id type x y z
1 1 4.0 0.0 0.0
2 2 10.0 10.0 10.0
ITEM: TIMESTEP
2
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 40.0
0.0 40.0
0.0 40.0
ITEM: ATOMS id type x y z
1 1 12.0 0.0 0.0
2 2 20.0 20.0 20.0
"""


# Same idea, but with a box tilted in all three directions (xy, xz, yz = 2, 1,
# 3 Å, doubling with the cell). Tilting xy alone is not enough to pin the
# orientation: with alpha = beta = 90 a rebuild from lengths and angles happens
# to land on the same vectors. Atom 1 sits at fractional (0.1, 0.1, 0.1) then
# (0.2, 0.1, 0.1), atom 2 at the centre.
LAMMPS_NPT_TRICLINIC_DUMP_FILE = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS xy xz yz pp pp pp
0.0 13.0 2.0
0.0 13.0 1.0
0.0 10.0 3.0
ITEM: ATOMS id type x y z
1 1 1.3 1.3 1.0
2 2 6.5 6.5 5.0
ITEM: TIMESTEP
1
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS xy xz yz pp pp pp
0.0 26.0 4.0
0.0 26.0 2.0
0.0 20.0 6.0
ITEM: ATOMS id type x y z
1 1 4.6 2.6 2.0
2 2 13.0 13.0 10.0
"""


@pytest.fixture
def lammps_npt_files(tmp_path):
    """Write a minimal NPT LAMMPS dump plus its data file to a temp dir."""
    data_file = tmp_path / 'lammps_data.txt'
    coords_file = tmp_path / 'lammps_npt.lammpstrj'
    data_file.write_text(LAMMPS_DATA_FILE)
    coords_file.write_text(LAMMPS_NPT_DUMP_FILE)
    return coords_file, data_file


def test_from_lammps_variable_lattice(lammps_npt_files):
    # issue #417: the per-frame box comes from the dump, not from the data file.
    coords_file, data_file = lammps_npt_files

    traj = Trajectory.from_lammps(
        coords_file=coords_file,
        data_file=data_file,
        coords_format='LAMMPSDUMP',
        temperature=300,
        time_step=1,
        type_mapping={'1': 'Li', '2': 'S'},
        constant_lattice=False,
    )

    assert traj.constant_lattice is False
    assert traj.species == [Element('Li'), Element('S')]
    assert traj.positions.shape == (3, 2, 3)

    # The box grows per frame; the static 10 Å box of the data file is ignored.
    assert_allclose(traj.lattice, [np.eye(3) * 10, np.eye(3) * 20, np.eye(3) * 40], atol=1e-4)

    assert_allclose(
        traj.positions,
        [
            [[0.1, 0.0, 0.0], [0.5, 0.5, 0.5]],
            [[0.2, 0.0, 0.0], [0.5, 0.5, 0.5]],
            [[0.3, 0.0, 0.0], [0.5, 0.5, 0.5]],
        ],
        atol=1e-6,
    )


def test_from_lammps_variable_lattice_distances_from_base_position(lammps_npt_files):
    # issue #417: each frame must be measured in its own (growing) cell, so
    # atom 1 moves 0.1 fractional per frame -> 0.1 * 20 Å and 0.2 * 40 Å.
    coords_file, data_file = lammps_npt_files

    traj = Trajectory.from_lammps(
        coords_file=coords_file,
        data_file=data_file,
        coords_format='LAMMPSDUMP',
        temperature=300,
        time_step=1,
        type_mapping={'1': 'Li', '2': 'S'},
        constant_lattice=False,
    )

    distances = traj.distances_from_base_position()

    assert distances.shape == (2, 3)
    assert_allclose(distances, [[0.0, 2.0, 8.0], [0.0, 0.0, 0.0]], atol=1e-6)


def test_from_lammps_variable_lattice_triclinic(tmp_path):
    # issue #417: a dump stores the xy/xz/yz tilt factors, so a tilted box is
    # fully defined. The cell must keep LAMMPS' orientation (a along x, b in
    # the x/y-plane) -- rebuilding it from lengths and angles rotates it away
    # from the frame the dump's cartesian coordinates are in, which silently
    # yields wrong fractional coordinates.
    data_file = tmp_path / 'lammps_data.txt'
    coords_file = tmp_path / 'lammps_npt_triclinic.lammpstrj'
    data_file.write_text(LAMMPS_DATA_FILE)
    coords_file.write_text(LAMMPS_NPT_TRICLINIC_DUMP_FILE)

    traj = Trajectory.from_lammps(
        coords_file=coords_file,
        data_file=data_file,
        coords_format='LAMMPSDUMP',
        temperature=300,
        time_step=1,
        type_mapping={'1': 'Li', '2': 'S'},
        constant_lattice=False,
    )

    assert_allclose(
        traj.lattice,
        [
            [[10, 0, 0], [2, 10, 0], [1, 3, 10]],
            [[20, 0, 0], [4, 20, 0], [2, 6, 20]],
        ],
        atol=1e-5,
    )

    assert_allclose(
        traj.positions,
        [
            [[0.1, 0.1, 0.1], [0.5, 0.5, 0.5]],
            [[0.2, 0.1, 0.1], [0.5, 0.5, 0.5]],
        ],
        atol=1e-6,
    )

    # Atom 1 moves 0.1 along a, measured in the second frame's 20 Å a vector.
    assert_allclose(traj.distances_from_base_position(), [[0.0, 2.0], [0.0, 0.0]], atol=1e-5)


def test_from_lammps_variable_lattice_needs_a_box_in_the_coords_file(tmp_path):
    # issue #417: xyz carries no box, so there is nothing to build the
    # per-frame lattice from -- say so instead of silently using a wrong cell.
    data_file = tmp_path / 'lammps_data.txt'
    coords_file = tmp_path / 'lammps_coords.xyz'
    data_file.write_text(LAMMPS_DATA_FILE)
    coords_file.write_text('2\nAtoms. Timestep: 0\nLi 1.0 0.0 0.0\nS 5.0 5.0 5.0\n')

    with pytest.raises(ValueError, match='contains no box'):
        Trajectory.from_lammps(
            coords_file=coords_file,
            data_file=data_file,
            temperature=300,
            time_step=1,
            constant_lattice=False,
        )


def test_mean_squared_displacement_variable_matches_constant():
    # When every frame's lattice is identical, the per-frame path must match the
    # constant-lattice path exactly.
    coords = np.array(
        [
            [[0.2, 0.0, 0.0], [0.0, 0.0, 0.5]],
            [[0.4, 0.1, 0.0], [0.0, 0.0, 0.5]],
            [[0.6, 0.2, 0.1], [0.0, 0.0, 0.5]],
        ]
    )
    lattice = np.diag([2.0, 3.0, 4.0])

    kwargs = dict(
        species=[Species('Li'), Species('S')],
        coords=coords,
        metadata={'temperature': 123},
        time_step=1,
    )
    const = Trajectory(lattice=lattice, constant_lattice=True, **kwargs)
    var = Trajectory(
        lattice=np.array([lattice, lattice, lattice]),
        constant_lattice=False,
        **kwargs,
    )

    assert_allclose(var.mean_squared_displacement(), const.mean_squared_displacement())


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


def test_drift_floating_species(trajectory):
    # 'C' is the only floating species, so the fixed set is B, Si, S, matching
    # fixed_species=['B', 'Si', 'S']
    drift = trajectory.drift(floating_species='C')
    assert drift.shape == (5, 1, 3)
    assert not np.isnan(drift).any()
    assert_allclose(drift, trajectory.drift(fixed_species=['B', 'Si', 'S']))


def test_drift_floating_species_element(trajectory):
    # Trajectories from e.g. from_vasprun carry Element (not Species) sites;
    # drift(floating_species=...) must accept those too (issue #406).
    element_traj = Trajectory(
        species=[Element(sp.symbol) for sp in trajectory.species],
        coords=trajectory.positions,
        lattice=trajectory.lattice,
        metadata=trajectory.metadata,
        time_step=trajectory.time_step,
    )
    drift = element_traj.drift(floating_species='C')
    assert drift.shape == (5, 1, 3)
    assert_allclose(drift, trajectory.drift(floating_species='C'))


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
    assert trajectory.kinisi_cache_data['diffusion_analyzer'] is not None
    assert trajectory.kinisi_cache_data['cache_key'] is not None

    diff2 = trajectory.kinisi_cache_data['diffusion_analyzer']

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
    assert_allclose(dt.values, [1.0e12, 2.0e12, 3.0e12, 4.0e12])
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


def test_to_ase_trajectory(trajectory):
    ase_traj = trajectory.to_ase_trajectory()

    assert np.all(
        ase_traj[3].positions
        == np.array([[0.8, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.5], [0.0, 0.0, 0.5]])
    )
    assert len(ase_traj) == 5
    assert np.all(ase_traj[0].get_atomic_numbers() == np.array([5, 14, 16, 6]))


def test_from_ase_trajectory(trajectory):
    ase_traj = trajectory.to_ase_trajectory()
    traj = Trajectory.from_ase_trajectory(
        trajectory=ase_traj,
        constant_lattice=True,
        temperature=123,
        time_step_ps=1e12,
    )

    assert isinstance(traj, Trajectory)
    assert traj.species == [
        Species('B'),
        Species('Si'),
        Species('S'),
        Species('C'),
    ]
    assert traj.positions.shape == (5, 4, 3)
    assert traj.metadata == {'temperature': 123}
    assert traj.time_step == 1
