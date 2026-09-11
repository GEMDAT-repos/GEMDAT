from __future__ import annotations

from dataclasses import fields, replace
from unittest.mock import patch

import numpy as np
import pytest
from pymatgen.core import Species, Structure

from gemdat.crystallizer import Crystallizer, CrystallizerResult, CrystallizerScan
from gemdat.io import read_cif
from gemdat.symmetry import SymmetryAnalyzer, SymmetryLevel, SymmetryRanking
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
    # the automatic scan reports the tolerance the winning group requires
    assert 0.01 <= result.symprec <= 0.5


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

    assert result.has_partial_occupancies is False
    occupancies = _mobile_occupancies(result.structure)
    assert len(occupancies) > 0
    assert all(occ == 1.0 for occ in occupancies)


def test_crystallize_use_density_default_unchanged(crystal_trajectory):
    # Baseline captured from the pre-change code on this fixture (default
    # resolution 0.2): the highest space group is C2/m (12), and the mobile
    # sites end up with partial occupancy. The tolerance it is reached at is
    # measured by the scan, so it is not pinned here.
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li')

    for result in (cr.crystallize(), cr.crystallize(use_density=True)):
        assert result.has_partial_occupancies is True
        assert result.spacegroup_number == 12

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


def test_scan_use_density_false(crystal_trajectory):
    # the toggle rides along on the scan, so every fit picked off it drops the
    # occupancies too
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li')

    scan = cr.scan(use_density=False)

    for result in (scan.best(), scan.at(0.1)):
        assert result.has_partial_occupancies is False
        occupancies = _mobile_occupancies(result.structure)
        assert len(occupancies) > 0
        assert all(occ == 1.0 for occ in occupancies)


def test_crystallize_at_use_density_false(crystal_trajectory):
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li')

    result = cr.crystallize_at(0.1, use_density=False)

    assert result.has_partial_occupancies is False
    occupancies = _mobile_occupancies(result.structure)
    assert len(occupancies) > 0
    assert all(occ == 1.0 for occ in occupancies)


def test_use_density_does_not_change_the_symmetry(crystal_trajectory):
    # only the occupancy weighting is dropped; the sites are located the same
    # way either way, so the fit must be identical
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li')

    with_density = cr.crystallize()
    without_density = cr.crystallize(use_density=False)

    assert without_density.spacegroup_number == with_density.spacegroup_number
    assert without_density.symprec == with_density.symprec


def test_use_density_does_not_invalidate_the_geometry_cache(crystal_trajectory):
    # the flag is applied to the occupancies after the cache, so toggling it
    # must not re-extract the density peaks
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li', resolution=0.5)

    with patch.object(
        Crystallizer,
        '_compute_geometry_and_occupancies',
        autospec=True,
        side_effect=Crystallizer._compute_geometry_and_occupancies,
    ) as compute:
        cr.crystallize_at(0.3)
        cr.crystallize_at(0.3, use_density=False)
        assert compute.call_count == 1


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


SYMPREC_RANGE = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5)


def test_scan_at_lists_the_given_tolerances(crystal_trajectory):
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li', resolution=0.5)

    ranking = cr.scan_at(SYMPREC_RANGE).ranking

    assert [level.symprec for level in ranking] == sorted(SYMPREC_RANGE)
    assert all(isinstance(level, SymmetryLevel) for level in ranking)
    for level in ranking:
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
    assert ranking[0].spacegroup_number == 1
    assert ranking[-1].spacegroup_number > 1


def test_scan_ranks_by_default(crystal_trajectory):
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li', resolution=0.5)

    scan = cr.scan(n_samples=20)

    assert isinstance(scan, CrystallizerScan)
    # the sweep itself is covered in tests/symmetry_test.py; here it only has to
    # arrive with the reconstructed geometry
    assert isinstance(scan.ranking, SymmetryRanking)
    assert len(scan.ranking) >= 1
    assert all(level.deviation is not None for level in scan.ranking)
    assert scan.ranking[0].spacegroup_number == 1


def test_scan_candidates_and_best(crystal_trajectory):
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li', resolution=0.5)

    scan = cr.scan(n_samples=20)

    # the sweep already yields one level per group, so candidates is the same
    # set, just ranked by space-group number instead of by deviation
    assert {c.spacegroup_number for c in scan.candidates} == {
        level.spacegroup_number for level in scan.ranking
    }
    numbers = [c.spacegroup_number for c in scan.candidates]
    assert numbers == sorted(numbers, reverse=True)
    # the winner is the highest-symmetry candidate
    assert scan.best().spacegroup_number == numbers[0]


def test_scan_at_candidates(crystal_trajectory):
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li', resolution=0.5)

    scan = cr.scan_at(SYMPREC_RANGE)
    ranking = scan.ranking

    assert len(ranking) == len(SYMPREC_RANGE)
    assert scan.candidates == ranking.candidates

    numbers = [c.spacegroup_number for c in scan.candidates]
    assert numbers == sorted(numbers, reverse=True)
    assert len(numbers) == len(set(numbers))
    # the winning space group is the highest-numbered candidate
    assert numbers[0] == scan.best().spacegroup_number
    # each candidate is the tightest symprec producing that space group
    for cand in scan.candidates:
        tighter = [
            level
            for level in ranking
            if level.symprec < cand.symprec
            and level.spacegroup_number == cand.spacegroup_number
        ]
        assert not tighter


def test_crystallize_at_skips_the_scan(crystal_trajectory):
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li', resolution=0.5)

    with patch.object(SymmetryAnalyzer, 'rank', autospec=True) as rank:
        result = cr.crystallize_at(0.3)

    assert result.symprec == 0.3
    # the tolerance was given, so no sweep was run to find one
    rank.assert_not_called()


def test_result_is_a_plain_value(crystal_trajectory):
    # a result holds no scan: another space group is asked of the scan it came
    # from, never of the result
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li', resolution=0.5)

    result = cr.crystallize_at(0.3)

    assert {field.name for field in fields(result)} == {
        'structure',
        'spacegroup_symbol',
        'spacegroup_number',
        'symprec',
        'has_partial_occupancies',
    }


def test_at_spacegroup_number(crystal_trajectory):
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li', resolution=0.5)

    scan = cr.scan(n_samples=20)
    target = scan.best().spacegroup_number

    result = scan.at_spacegroup(target)

    assert result.spacegroup_number == target
    # tightest symprec in the scan that still yields the target
    tighter = [
        level
        for level in scan.ranking
        if level.symprec < result.symprec and level.spacegroup_number == target
    ]
    assert not tighter


def test_at_spacegroup_symbol(crystal_trajectory):
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li', resolution=0.5)

    scan = cr.scan(n_samples=20)
    baseline = scan.best()
    # match tolerant of case/whitespace
    messy = f'  {baseline.spacegroup_symbol.lower()} '

    result = scan.at_spacegroup(messy)

    assert result.spacegroup_number == baseline.spacegroup_number


def test_at_spacegroup_unreachable(crystal_trajectory):
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li', resolution=0.5)

    scan = cr.scan(n_samples=20)

    with pytest.raises(ValueError, match='space group 216') as excinfo:
        scan.at_spacegroup(216)

    # the error lists what was actually found (the ranking table)
    assert 'symprec (Å)' in str(excinfo.value)


def test_scan_without_any_symmetry_raises(crystal_trajectory):
    # 0.0 is not a usable tolerance, so every fit in the sweep fails
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li', resolution=0.5)

    scan = cr.scan_at((0.0,))

    with pytest.raises(ValueError, match='Could not determine symmetry for any'):
        scan.best()


def test_crystallize_angle_tolerance_is_used(crystal_trajectory):
    # a negative angle tolerance switches spglib to its own algorithm rather
    # than being an error, so check the value arrives at the analyzer
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li', resolution=0.5)

    seen = []
    original = SymmetryAnalyzer.__init__

    def record(self, structure, *, angle_tolerance=5.0):
        seen.append(angle_tolerance)
        original(self, structure, angle_tolerance=angle_tolerance)

    with patch.object(SymmetryAnalyzer, '__init__', record):
        cr.crystallize(n_samples=5, angle_tolerance=1.0)
        cr.scan_at(SYMPREC_RANGE, angle_tolerance=2.0)

    assert seen == [1.0, 2.0]


def test_geometry_is_computed_once_per_argument_set(crystal_trajectory):
    # peak extraction dominates the cost, so repeated calls must reuse it
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li', resolution=0.5)

    with patch.object(
        Crystallizer,
        '_compute_geometry_and_occupancies',
        autospec=True,
        side_effect=Crystallizer._compute_geometry_and_occupancies,
    ) as compute:
        cr.crystallize_at(0.3)
        cr.crystallize_at(0.3)
        cr.scan_at(SYMPREC_RANGE)
        assert compute.call_count == 1

        # different arguments are a different geometry
        cr.crystallize_at(0.3, background_level=0.2)
        assert compute.call_count == 2

        # unhashable arguments simply bypass the cache (voxel coordinates of
        # the two Li sites on the 20^3 grid this resolution gives)
        peaks = np.array([[5, 5, 5], [15, 15, 15]])
        cr.scan_at(SYMPREC_RANGE, peaks=peaks)
        cr.scan_at(SYMPREC_RANGE, peaks=peaks)
        assert compute.call_count == 4


def test_crystallize_at_failure_raises(crystal_trajectory):
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li', resolution=0.5)

    # 0.0 is not a usable tolerance for the symmetry finder
    with pytest.raises(ValueError, match='symprec=0.0'):
        cr.crystallize_at(0.0)


def test_scan_format(crystal_trajectory):
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li', resolution=0.5)

    table = cr.scan_at(SYMPREC_RANGE).format()

    assert isinstance(table, str)
    lines = table.splitlines()
    assert 'symprec (Å)' in lines[0]
    assert 'space group' in lines[0]
    assert '# orbits' in lines[0]
    # header + separator + one row per symprec
    assert len(lines) == 2 + len(SYMPREC_RANGE)
    for symprec in SYMPREC_RANGE:
        assert any(line.startswith(f'{symprec:g}') for line in lines[2:])


def test_result_to_cif(crystal_trajectory, tmp_path):
    # a result that has already been inspected writes itself, without refitting
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li', resolution=0.5)
    result = cr.crystallize(n_samples=20)

    filename = tmp_path / 'crystallized.cif'
    result.to_cif(filename)

    assert filename.exists()
    assert isinstance(read_cif(filename), Structure)


def test_at_level(crystal_trajectory):
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li', resolution=0.5)
    scan = cr.scan_at(SYMPREC_RANGE)

    # the lowest-symmetry row of the ranking, i.e. not the one `best` picks
    level = min(scan.ranking.found, key=lambda level: level.spacegroup_number)

    result = scan.at_level(level)

    assert result.spacegroup_number == level.spacegroup_number
    assert result.spacegroup_symbol == level.spacegroup_symbol
    assert result.symprec == level.symprec


def test_to_cif_per_candidate(crystal_trajectory, tmp_path):
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li', resolution=0.5)
    scan = cr.scan_at(SYMPREC_RANGE)

    for level in scan.candidates:
        filename = tmp_path / f'{level.spacegroup_number}.cif'
        result = scan.at_level(level)
        result.to_cif(filename)

        assert filename.exists()
        assert result.spacegroup_number == level.spacegroup_number
        assert isinstance(read_cif(filename), Structure)


def test_at_level_from_other_scan_raises(crystal_trajectory):
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li', resolution=0.5)
    scan = cr.scan_at(SYMPREC_RANGE)
    level = scan.ranking.best()

    # a level whose space group this geometry does not reproduce cannot have
    # come from a ranking of it
    bogus = replace(level, spacegroup_number=216, spacegroup_symbol='F-43m')

    with pytest.raises(ValueError, match='not part of this scan'):
        scan.at_level(bogus)


def test_at_level_failed_raises(crystal_trajectory):
    cr = Crystallizer.from_trajectory(crystal_trajectory, floating_specie='Li', resolution=0.5)

    # 0.0 is not a usable tolerance, so the only row of this scan failed
    scan = cr.scan_at((0.0,))
    (failed,) = scan.ranking

    assert failed.error is not None

    with pytest.raises(ValueError, match='found no space group'):
        scan.at_level(failed)
