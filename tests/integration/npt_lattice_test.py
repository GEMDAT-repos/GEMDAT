"""Regression tests for variable-lattice (NPT) trajectories, see issue #394.

These exercise the lattice-handling fixes on a real NPT vasprun (the
first 200 ionic steps of a Li6PS5Cl run with ``ISIF=3``, so the cell
varies per frame), loaded with ``constant_lattice=False``.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.npt_vaspxml_available  # type: ignore
def test_npt_load(npt_traj):
    assert len(npt_traj) == 200
    assert len(npt_traj.species) == 416
    assert npt_traj.constant_lattice is False

    lattice = np.asarray(npt_traj.lattice)
    assert lattice.shape == (200, 3, 3)

    # An NPT run has a genuinely varying cell, otherwise this is not a
    # meaningful test of the variable-lattice code path.
    assert not np.allclose(lattice[0], lattice[-1])


@pytest.npt_vaspxml_available  # type: ignore
def test_npt_get_lattice(npt_traj):
    # With an index, get_lattice() returns that frame's cell.
    first = npt_traj.get_lattice(0)
    last = npt_traj.get_lattice(-1)
    assert np.isclose(first.abc[0], 20.32466211)
    assert np.isclose(last.abc[0], 19.844281199922083)
    assert not np.isclose(first.volume, last.volume)

    # Without an index the cell is ambiguous -> raise instead of crashing
    # (issue #394).
    with pytest.raises(ValueError, match='not constant'):
        npt_traj.get_lattice()


@pytest.npt_vaspxml_available  # type: ignore
def test_npt_filter_preserves_per_frame_lattice(npt_traj):
    # issue #394: filter() must forward the per-frame lattice.
    diff = npt_traj.filter('Li')

    assert len(diff.species) == 192
    assert all(sp.symbol == 'Li' for sp in diff.species)
    assert diff.constant_lattice is False
    assert np.allclose(np.asarray(diff.lattice), np.asarray(npt_traj.lattice))


@pytest.npt_vaspxml_available  # type: ignore
def test_npt_mean_squared_displacement(npt_traj):
    # issue #394: MSD used to call get_lattice() without an index and crash.
    msd = npt_traj.mean_squared_displacement()

    assert msd.shape == (416, 200)
    assert np.isclose(msd[0, -1], 0.4309191659960344)
    assert np.isclose(msd[100, -1], 3.7168323480795014)
