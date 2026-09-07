from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Species, Structure

from gemdat.crystallizer import Crystallizer, CrystallizerResult
from gemdat.io import read_cif
from gemdat.trajectory import Trajectory


@pytest.fixture()
def crystal_trajectory():
    """A well-sampled toy trajectory: two mobile Li sites that are visited
    every frame, plus a static P/S framework.

    Sampled densely enough that the density peak detection finds the
    mobile sites.
    """
    rng = np.random.default_rng(0)
    n_frames = 200

    li_sites = np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
    framework = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]])

    coords = np.empty((n_frames, 4, 3))
    for frame in range(n_frames):
        coords[frame, 0, :] = li_sites[frame % 2] + rng.normal(scale=0.01, size=3)
        coords[frame, 1, :] = li_sites[(frame + 1) % 2] + rng.normal(scale=0.01, size=3)
        coords[frame, 2, :] = framework[0] + rng.normal(scale=0.005, size=3)
        coords[frame, 3, :] = framework[1] + rng.normal(scale=0.005, size=3)
    coords %= 1

    return Trajectory(
        species=[Species('Li'), Species('Li'), Species('P'), Species('S')],
        coords=coords,
        lattice=np.eye(3) * 10.0,
        metadata={'temperature': 300},
        time_step=1,
    )


def test_framework(trajectory):
    # mobile species in the shared fixture is 'B'; framework = Si, S, C
    cr = Crystallizer.from_trajectory(trajectory, floating_specie='B')

    framework = cr.framework()

    assert isinstance(framework, Structure)
    assert len(framework) == 3
    assert {site.specie.symbol for site in framework} == {'Si', 'S', 'C'}


def test_mobile_sites_occupancies(crystal_trajectory):
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li', resolution=0.5)

    mobile = cr.mobile_sites()

    assert isinstance(mobile, Structure)
    assert len(mobile) > 0
    for site in mobile:
        occupancy = site.species.num_atoms
        assert 0 < occupancy <= 1.0


def test_crystallize(crystal_trajectory):
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li', resolution=0.5)

    result = cr.crystallize()

    assert isinstance(result, CrystallizerResult)
    assert isinstance(result.structure, Structure)
    assert len(result.structure) > 0
    assert result.spacegroup_number >= 1
    assert result.symprec in (0.01, 0.05, 0.1, 0.2, 0.3, 0.5)


def test_crystallize_empty_framework(crystal_trajectory):
    # A trajectory holding only the floating specie (e.g. already filtered with
    # `trajectory.filter('Li')`) has no static framework. Crystallizing it must
    # not crash on the empty framework, and should yield only mobile sites.
    mobile_only = crystal_trajectory.filter('Li')
    cr = Crystallizer.from_trajectory(mobile_only, floating_specie='Li', resolution=0.5)

    result = cr.crystallize()

    assert isinstance(result, CrystallizerResult)
    assert len(result.structure) > 0
    assert {next(iter(site.species.as_dict())) for site in result.structure} == {'Li'}


def test_to_cif(crystal_trajectory, tmp_path):
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li', resolution=0.5)

    filename = tmp_path / 'crystallized.cif'
    cr.to_cif(filename)

    assert filename.exists()

    reread = read_cif(filename)
    assert isinstance(reread, Structure)
    assert len(reread) > 0
    # symmetry was written (more than just P1 with the asymmetric unit)
    assert '_symmetry_space_group_name_H-M' in filename.read_text()


def _mobile_occupancies(structure, specie='Li'):
    """Occupancies of the mobile-species sites in a (crystallized)
    structure."""
    return [site.species.num_atoms for site in structure if specie in site.species.as_dict()]


def test_crystallize_use_density_false(crystal_trajectory):
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li')

    result = cr.crystallize(use_density=False)

    assert result.use_density is False
    occupancies = _mobile_occupancies(result.structure)
    assert len(occupancies) > 0
    assert all(occ == 1.0 for occ in occupancies)


def test_crystallize_use_density_default_unchanged(crystal_trajectory):
    # Baseline captured from the pre-change code on this fixture (default
    # resolution 0.2): highest space group is C2/m (12) at symprec 0.1, and
    # the mobile sites end up with partial occupancy.
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li')

    for result in (cr.crystallize(), cr.crystallize(use_density=True)):
        assert result.use_density is True
        assert result.spacegroup_number == 12
        assert result.symprec == 0.1

        occupancies = _mobile_occupancies(result.structure)
        assert len(occupancies) > 0
        assert any(occ < 1.0 for occ in occupancies)
        assert all(0 < occ <= 1.0 for occ in occupancies)


def test_to_cif_use_density_false(crystal_trajectory, tmp_path):
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li')

    filename = tmp_path / 'crystallized_no_density.cif'
    cr.to_cif(filename, use_density=False)

    assert filename.exists()

    reread = read_cif(filename)
    occupancies = _mobile_occupancies(reread)
    assert len(occupancies) > 0
    assert all(occ == 1.0 for occ in occupancies)


def test_mobile_sites_with_occupancies_false(crystal_trajectory):
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li', resolution=0.2)

    mobile = cr.mobile_sites(with_occupancies=False)

    assert isinstance(mobile, Structure)
    assert len(mobile) > 0
    for site in mobile:
        assert 'Li' in site.species.as_dict()
        assert site.species.num_atoms == 1.0


def test_framework_rejects_variable_lattice(variable_lattice_trajectory):
    crystallizer = Crystallizer(trajectory=variable_lattice_trajectory, floating_specie='Li')

    with pytest.raises(NotImplementedError, match='variable lattice'):
        crystallizer.framework()

    with pytest.raises(NotImplementedError, match='variable lattice'):
        crystallizer.crystallize()
