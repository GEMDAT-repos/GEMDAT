from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Species, Structure

from gemdat.crystallizer import Crystallizer, CrystallizerResult, SymmetryLevel
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


def test_framework_rejects_variable_lattice(variable_lattice_trajectory):
    crystallizer = Crystallizer(trajectory=variable_lattice_trajectory, floating_specie='Li')

    with pytest.raises(NotImplementedError, match='variable lattice'):
        crystallizer.framework()

    with pytest.raises(NotImplementedError, match='variable lattice'):
        crystallizer.crystallize()


SYMPREC_RANGE = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5)


def test_symmetry_ladder(crystal_trajectory):
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li', resolution=0.5)

    ladder = cr.symmetry_ladder(symprec_range=SYMPREC_RANGE)

    assert [level.symprec for level in ladder] == sorted(SYMPREC_RANGE)
    assert all(isinstance(level, SymmetryLevel) for level in ladder)
    for level in ladder:
        if level.error is None:
            assert isinstance(level.spacegroup_number, int)
            assert isinstance(level.spacegroup_symbol, str)
            assert isinstance(level.crystal_system, str)
            assert isinstance(level.n_symmetry_ops, int)
            assert isinstance(level.n_site_orbits, int)
        else:
            assert level.spacegroup_number is None

    # This toy fixture climbs from P1 (#1) at the tightest tolerance to a
    # higher-symmetry monoclinic cell once symprec is loosened.
    assert ladder[0].spacegroup_number == 1
    assert ladder[-1].spacegroup_number > 1


def test_crystallize_ladder_and_candidates(crystal_trajectory):
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li', resolution=0.5)

    result = cr.crystallize(symprec_range=SYMPREC_RANGE)

    assert result.ladder is not None
    assert len(result.ladder) == len(SYMPREC_RANGE)

    assert result.candidates is not None
    numbers = [c.spacegroup_number for c in result.candidates]
    assert numbers == sorted(numbers, reverse=True)
    assert len(numbers) == len(set(numbers))
    # the winning space group is the highest-numbered candidate
    assert numbers[0] == result.spacegroup_number
    # each candidate is the tightest symprec producing that space group
    for cand in result.candidates:
        tighter = [
            level
            for level in result.ladder
            if level.symprec < cand.symprec
            and level.spacegroup_number == cand.spacegroup_number
        ]
        assert not tighter


def test_crystallize_explicit_symprec_skips_sweep(crystal_trajectory):
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li', resolution=0.5)

    result = cr.crystallize(symprec=0.3)

    assert result.symprec == 0.3
    # no sweep was run
    assert result.ladder is None
    assert result.candidates is None


def test_crystallize_target_spacegroup_number(crystal_trajectory):
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li', resolution=0.5)

    baseline = cr.crystallize(symprec_range=SYMPREC_RANGE)
    target = baseline.spacegroup_number

    result = cr.crystallize(symprec_range=SYMPREC_RANGE, target_spacegroup=target)

    assert result.spacegroup_number == target
    # tightest symprec in the range that still yields the target
    assert result.ladder is not None
    tighter = [
        level
        for level in result.ladder
        if level.symprec < result.symprec and level.spacegroup_number == target
    ]
    assert not tighter


def test_crystallize_target_spacegroup_symbol(crystal_trajectory):
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li', resolution=0.5)

    baseline = cr.crystallize(symprec_range=SYMPREC_RANGE)
    # match tolerant of case/whitespace
    messy = f'  {baseline.spacegroup_symbol.lower()} '

    result = cr.crystallize(symprec_range=SYMPREC_RANGE, target_spacegroup=messy)

    assert result.spacegroup_number == baseline.spacegroup_number


def test_crystallize_target_spacegroup_unreachable(crystal_trajectory):
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li', resolution=0.5)

    with pytest.raises(ValueError, match='space group 216'):
        cr.crystallize(symprec_range=SYMPREC_RANGE, target_spacegroup=216)

    # the error lists what was actually found (the ladder table)
    try:
        cr.crystallize(symprec_range=SYMPREC_RANGE, target_spacegroup=216)
    except ValueError as exc:
        assert 'symprec (Å)' in str(exc)


def test_crystallize_symprec_and_target_are_contradictory(crystal_trajectory):
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li', resolution=0.5)

    with pytest.raises(ValueError, match='not both'):
        cr.crystallize(symprec=0.1, target_spacegroup=200)


def test_crystallize_explicit_symprec_failure_raises(crystal_trajectory):
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li', resolution=0.5)

    # 0.0 is not a usable tolerance for the symmetry finder
    with pytest.raises(ValueError, match='symprec=0.0'):
        cr.crystallize(symprec=0.0)


def test_format_ladder(crystal_trajectory):
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li', resolution=0.5)

    result = cr.crystallize(symprec_range=SYMPREC_RANGE)
    table = result.format_ladder()

    assert isinstance(table, str)
    lines = table.splitlines()
    assert 'symprec (Å)' in lines[0]
    assert 'space group' in lines[0]
    assert '# orbits' in lines[0]
    # header + separator + one row per symprec
    assert len(lines) == 2 + len(SYMPREC_RANGE)
    for symprec in SYMPREC_RANGE:
        assert any(line.startswith(f'{symprec:g}') for line in lines[2:])


def test_format_ladder_without_ladder_raises(crystal_trajectory):
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li', resolution=0.5)

    result = cr.crystallize(symprec=0.3)

    with pytest.raises(ValueError, match='no symmetry ladder'):
        result.format_ladder()
