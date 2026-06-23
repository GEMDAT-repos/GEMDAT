from __future__ import annotations

import pytest

from gemdat.trajectory import Trajectory

from .conftest import VASP_XML


@pytest.vaspxml_available  # type: ignore
def test_constant_lattice_cache_invalidation(tmp_path):
    """Changing `constant_lattice` must not reuse a stale cache (issue
    #393)."""
    # Symlink avoids copying the large vasprun.xml while keeping the
    # auto-generated cache files inside tmp_path.
    xml = tmp_path / 'vasprun.xml'
    xml.symlink_to(VASP_XML)

    traj_false = Trajectory.from_vasprun(xml, constant_lattice=False)
    assert traj_false.constant_lattice is False

    traj_true = Trajectory.from_vasprun(xml, constant_lattice=True)
    assert traj_true.constant_lattice is True

    # Each `constant_lattice` value should produce its own cache file.
    caches = sorted(tmp_path.glob('vasprun.xml.*.cache'))
    assert len(caches) == 2
