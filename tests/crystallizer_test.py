from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Species, Structure

from gemdat.crystallizer import Crystallizer, CrystallizerResult
from gemdat.io import load_known_material, read_cif
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


@pytest.fixture()
def ideal_argyrodite_trajectory():
    """Synthetic trajectory of the *ideal* argyrodite Li 48h sublattice in an
    exact cubic cell.

    Li frac coords and lattice are taken from the bundled ``argyrodite.cif``
    (F-43m, #216, Li 48h only). Every atom vibrates with tiny gaussian noise
    around its ideal Wyckoff position. There is no framework (Li only), so
    this also exercises the empty-framework path (see
    ``test_crystallize_empty_framework``).

    Unlike real Li6PS5Br MD - where the S/Br host averages to P1 and the Li
    density-peak centroids scatter ~0.3 A off the ideal 48h positions, so the
    crystallizer correctly returns P1 - this genuinely ideal input must
    crystallize back to the cubic argyrodite space group.
    """
    reference = load_known_material('argyrodite')
    li_frac = np.array([site.frac_coords for site in reference if site.specie.symbol == 'Li'])

    rng = np.random.default_rng(0)
    n_frames = 150
    n_atoms = len(li_frac)

    coords = np.empty((n_frames, n_atoms, 3))
    for frame in range(n_frames):
        coords[frame] = li_frac + rng.normal(scale=0.005, size=(n_atoms, 3))
    coords %= 1

    return Trajectory(
        species=[Species('Li')] * n_atoms,
        coords=coords,
        lattice=reference.lattice.matrix,
        metadata={'temperature': 300},
        time_step=1,
    )


def test_crystallize_ideal_argyrodite_sublattice(ideal_argyrodite_trajectory):
    """The symmetry machinery recovers argyrodite symmetry when the input
    genuinely has it (contrast with the real-MD integration test, which is
    expected to yield P1)."""
    cr = Crystallizer.from_trajectory(
        ideal_argyrodite_trajectory, floating_specie='Li', resolution=0.2
    )

    result = cr.crystallize()

    # One mobile site per ideal 48h Wyckoff position.
    li_sites = [site for site in result.structure if 'Li' in site.species.as_dict()]
    assert len(li_sites) == 48

    # A cubic space group (number >= 195) is the hard requirement. With this
    # little noise the density path recovers F-43m (#216) exactly, so assert
    # that too - a drop below it is a real regression worth seeing.
    assert result.spacegroup_number >= 195
    assert result.spacegroup_number == 216
    assert result.spacegroup_symbol == 'F-43m'


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


def test_framework_rejects_variable_lattice(variable_lattice_trajectory):
    crystallizer = Crystallizer(trajectory=variable_lattice_trajectory, floating_specie='Li')

    with pytest.raises(NotImplementedError, match='variable lattice'):
        crystallizer.framework()

    with pytest.raises(NotImplementedError, match='variable lattice'):
        crystallizer.crystallize()
