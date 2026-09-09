from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from gemdat.symmetry import SymmetryAnalyzer, SymmetryLevel, SymmetryRanking, _rank_key

SYMPREC_RANGE = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5)


def _structure(perturb: float | None = None) -> Structure:
    """A small P/S framework with two Li sites, optionally jittered so that it
    only reaches its symmetry once the tolerance is loosened."""
    structure = Structure(
        lattice=Lattice.cubic(10.0),
        species=['P', 'S', 'Li', 'Li'],
        coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
    )
    if perturb is not None:
        structure.perturb(perturb, seed=0)
    return structure


@pytest.fixture()
def ideal_structure():
    return _structure()


@pytest.fixture()
def noisy_structure():
    return _structure(perturb=0.05)


@pytest.fixture()
def sheared_structure():
    """A cubic-content cell whose gamma angle is off by exactly 2 degrees, so
    that the angular cost of fitting it as cubic is known analytically."""
    return Structure(
        lattice=Lattice.from_parameters(10.0, 10.0, 10.0, 90.0, 90.0, 92.0),
        species=['Li', 'Cl'],
        coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def _level(number, symbol, symprec=0.1, **kwargs):
    return SymmetryLevel(
        symprec=symprec,
        spacegroup_number=number,
        spacegroup_symbol=symbol,
        crystal_system='cubic',
        n_symmetry_ops=1,
        n_site_orbits=1,
        **kwargs,
    )


def test_level(ideal_structure):
    level = SymmetryAnalyzer(ideal_structure)._level(0.1)

    assert isinstance(level, SymmetryLevel)
    assert level.symprec == 0.1
    assert isinstance(level.spacegroup_number, int)
    assert isinstance(level.spacegroup_symbol, str)
    assert isinstance(level.crystal_system, str)
    assert isinstance(level.n_symmetry_ops, int)
    assert isinstance(level.n_site_orbits, int)
    assert level.error is None
    # measuring the deviation is opt-in
    assert level.deviation is None
    assert level.angle_deviation is None


def test_level_records_failure_instead_of_raising(ideal_structure):
    level = SymmetryAnalyzer(ideal_structure)._level(0.0)

    assert level.symprec == 0.0
    assert level.spacegroup_number is None
    assert level.error


def test_level_with_deviation(ideal_structure):
    level = SymmetryAnalyzer(ideal_structure)._level(0.1, with_deviation=True)

    # an undistorted structure satisfies its own symmetry exactly
    assert level.deviation == pytest.approx(0.0, abs=1e-9)
    assert level.angle_deviation == pytest.approx(0.0, abs=1e-9)


def test_deviation_is_the_displacement_the_group_demands():
    # Cl sits 0.2 A off the body centre along x. The cubic group contains
    # x -> -x about the origin, which maps it onto its mirror image 0.4 A
    # away, so that is exactly what the fit costs.
    structure = Structure(
        lattice=Lattice.cubic(10.0),
        species=['Li', 'Cl'],
        coords=[[0.0, 0.0, 0.0], [0.52, 0.5, 0.5]],
    )

    level = SymmetryAnalyzer(structure)._level(0.5, with_deviation=True)

    assert level.spacegroup_number == 221
    assert level.deviation == pytest.approx(0.4)
    assert level.angle_deviation == pytest.approx(0.0, abs=1e-9)


def test_angle_deviation_is_the_angle_the_group_demands(sheared_structure):
    # fitting a gamma = 92 deg cell as cubic idealises gamma back to 90 deg
    level = SymmetryAnalyzer(sheared_structure)._level(0.1, with_deviation=True)

    assert level.spacegroup_number == 221
    assert level.angle_deviation == pytest.approx(2.0)
    # the atoms themselves do not have to move for it
    assert level.deviation == pytest.approx(0.0, abs=1e-9)


def test_angle_tolerance_is_honoured(sheared_structure):
    # 2 deg of shear is within the default tolerance, but not within 1 deg
    loose = SymmetryAnalyzer(sheared_structure, angle_tolerance=5.0)._level(0.1)
    tight = SymmetryAnalyzer(sheared_structure, angle_tolerance=1.0)._level(0.1)

    assert loose.spacegroup_number == 221
    assert tight.spacegroup_number is not None
    assert tight.spacegroup_number < loose.spacegroup_number


def test_deviation_matches_sites_one_to_one():
    # A degenerate operation that collapses both sites onto one point: taking
    # each image's nearest site would report 0.2 A (both images land next to
    # Li), but an operation permutes the sites, and no permutation does better
    # than the 2.8 A of moving one image onto the second site.
    structure = Structure(
        lattice=Lattice.cubic(10.0),
        species=['Li', 'Li'],
        coords=[[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]],
    )
    dataset = SimpleNamespace(
        rotations=[np.zeros((3, 3), dtype=int)],
        translations=[np.array([0.02, 0.0, 0.0])],
        std_lattice=structure.lattice.matrix,
        std_rotation_matrix=np.eye(3),
        transformation_matrix=np.eye(3),
    )

    deviation, angle_deviation, error = SymmetryAnalyzer(structure)._deviation(dataset)

    assert error is None
    assert deviation == pytest.approx(2.8)
    assert angle_deviation == pytest.approx(0.0, abs=1e-9)


def test_deviation_failure_is_recorded_on_the_level(ideal_structure, monkeypatch):
    # the deviation is a diagnostic: a failure to measure it must not throw the
    # space group away, but must not pass silently either
    monkeypatch.setattr(
        'gemdat.symmetry.linear_sum_assignment',
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('boom')),
    )

    level = SymmetryAnalyzer(ideal_structure)._level(0.1, with_deviation=True)

    assert level.spacegroup_number is not None
    assert level.deviation is None
    assert level.angle_deviation is None
    assert 'boom' in level.error


def test_scan(noisy_structure):
    ranking = SymmetryAnalyzer(noisy_structure).rank(symprec_min=0.01, symprec_max=0.5)

    assert isinstance(ranking, SymmetryRanking)

    # one entry per distinct space group, least demanding first
    numbers = [level.spacegroup_number for level in ranking]
    assert len(numbers) == len(set(numbers))
    assert [level.deviation for level in ranking] == sorted(
        level.deviation for level in ranking
    )

    for level in ranking:
        assert 0.01 <= level.symprec <= 0.5
        assert level.error is None
        assert level.deviation is not None and level.deviation >= 0
        assert level.angle_deviation is not None and level.angle_deviation >= 0
        assert level.n_observed is not None and level.n_observed >= 1

    # P1 is always satisfied exactly, so it demands nothing; the jitter has to
    # be absorbed before any higher symmetry holds
    assert ranking[0].spacegroup_number == 1
    assert ranking[0].deviation == pytest.approx(0.0, abs=1e-9)
    assert all(level.deviation > 0 for level in ranking[1:])
    assert max(numbers) > 1


def test_scan_deviation_is_independent_of_symprec(noisy_structure):
    """The deviation is a property of the structure, so the tolerance that
    happened to find the group must not change it."""
    analyzer = SymmetryAnalyzer(noisy_structure)
    winner = analyzer.rank(symprec_min=0.01, symprec_max=0.5).best()
    assert winner.spacegroup_number is not None and winner.spacegroup_number > 1

    compared = 0
    for symprec in (winner.symprec, 0.3, 0.5):
        level = analyzer._level(symprec, with_deviation=True)
        if level.spacegroup_number != winner.spacegroup_number:
            continue
        assert level.deviation == pytest.approx(winner.deviation)
        assert level.angle_deviation == pytest.approx(winner.angle_deviation)
        compared += 1

    # the group need not turn up at every tolerance, but if it never did the
    # loop above would assert nothing at all
    assert compared


def test_scan_orders_unmeasurable_deviations_last(noisy_structure, monkeypatch):
    analyzer = SymmetryAnalyzer(noisy_structure)
    real_level = SymmetryAnalyzer._level

    def flaky(self, symprec, *, with_deviation=False):
        level = real_level(self, symprec, with_deviation=with_deviation)
        # P1 holds exactly, so it would otherwise sort first
        if with_deviation and level.spacegroup_number == 1:
            return replace(level, deviation=None, angle_deviation=None, error='no idea')
        return level

    monkeypatch.setattr(SymmetryAnalyzer, '_level', flaky)

    ranking = analyzer.rank(symprec_min=0.01, symprec_max=0.5)

    assert len(ranking) > 1
    assert ranking[-1].spacegroup_number == 1
    assert ranking[-1].deviation is None
    # the failure is reported rather than looking like "not measured"
    assert ranking[-1].error == 'no idea'


def test_scan_breaks_deviation_ties_on_the_angle():
    ranking = SymmetryRanking(
        sorted(
            [
                _level(12, 'C2/m', deviation=0.1, angle_deviation=1.0),
                _level(65, 'Cmmm', deviation=0.1, angle_deviation=0.2),
                _level(1, 'P1', deviation=0.0, angle_deviation=0.0),
            ],
            key=_rank_key,
        )
    )

    assert [level.spacegroup_number for level in ranking] == [1, 65, 12]


def test_scan_rejects_invalid_range(ideal_structure):
    analyzer = SymmetryAnalyzer(ideal_structure)

    with pytest.raises(ValueError, match='`symprec_min` must be positive'):
        analyzer.rank(symprec_min=0.0)

    with pytest.raises(ValueError, match='must not be smaller'):
        analyzer.rank(symprec_min=0.5, symprec_max=0.1)

    with pytest.raises(ValueError, match='`n_samples` must be at least 2'):
        analyzer.rank(n_samples=1)


def test_rank_dispatches_on_symprec_range(noisy_structure):
    analyzer = SymmetryAnalyzer(noisy_structure)

    # a fixed list of tolerances is swept as given...
    listed = analyzer.rank(symprec_range=SYMPREC_RANGE)
    assert [level.symprec for level in listed] == sorted(SYMPREC_RANGE)
    assert all(level.deviation is None for level in listed)

    # ...otherwise the range is scanned and the deviations measured
    scanned = analyzer.rank(n_samples=10)
    assert all(level.deviation is not None for level in scanned)


@pytest.mark.parametrize(
    'kwargs', [{'symprec_min': 0.02}, {'symprec_max': 0.4}, {'n_samples': 10}]
)
def test_rank_rejects_scan_settings_with_an_explicit_range(ideal_structure, kwargs):
    # silently ignoring the scan settings would hide a mistake
    analyzer = SymmetryAnalyzer(ideal_structure)

    with pytest.raises(ValueError, match='cannot be combined'):
        analyzer.rank(symprec_range=SYMPREC_RANGE, **kwargs)


def test_explicit_range_skips_deviation(noisy_structure):
    ranking = SymmetryAnalyzer(noisy_structure).rank(symprec_range=SYMPREC_RANGE)

    assert [level.symprec for level in ranking] == sorted(SYMPREC_RANGE)
    # measuring is operations x sites^2, so a plain sweep does not do it
    assert all(level.deviation is None for level in ranking)
    assert all(level.n_observed is None for level in ranking)


def test_ranking_is_a_sequence():
    levels = [_level(1, 'P1'), _level(12, 'C2/m')]

    ranking = SymmetryRanking(levels)

    assert len(ranking) == 2
    assert list(ranking) == levels
    assert ranking[0] is levels[0]
    assert ranking[-1] is levels[-1]
    assert 'C2/m (12)' in repr(ranking)


def test_ranking_found_drops_failures():
    failed = SymmetryLevel(symprec=0.0, error='nope')

    ranking = SymmetryRanking([failed, _level(12, 'C2/m')])

    assert ranking.found == [ranking[1]]


def test_ranking_candidates():
    ranking = SymmetryRanking(
        [
            _level(1, 'P1', symprec=0.01),
            _level(12, 'C2/m', symprec=0.1),
            _level(12, 'C2/m', symprec=0.2),
            SymmetryLevel(symprec=0.3, error='nope'),
        ]
    )

    candidates = ranking.candidates

    # one entry per group, highest first, each at its tightest tolerance
    assert [c.spacegroup_number for c in candidates] == [12, 1]
    assert [c.symprec for c in candidates] == [0.1, 0.01]


def test_ranking_match():
    ranking = SymmetryRanking(
        [_level(12, 'C2/m', symprec=0.2), _level(12, 'C2/m', symprec=0.1)]
    )

    # tightest tolerance reaching the target, by number or by symbol
    assert ranking.match(12).symprec == 0.1
    assert ranking.match('C2/m').symprec == 0.1
    # symbol matching ignores case and whitespace
    assert ranking.match('  c2/m ').symprec == 0.1

    assert ranking.match(216) is None
    assert ranking.match('F-43m') is None


def test_ranking_match_ignores_levels_without_a_symbol():
    # a level can carry a number but no symbol; matching by symbol must skip it
    # instead of tripping over the `None`
    ranking = SymmetryRanking([_level(12, None, symprec=0.1)])

    assert ranking.match('C2/m') is None
    assert ranking.match(12) is not None


def test_ranking_best_prefers_symmetry_then_tightness():
    ranking = SymmetryRanking(
        [
            _level(1, 'P1', symprec=0.01),
            _level(12, 'C2/m', symprec=0.3),
            _level(12, 'C2/m', symprec=0.1),
        ]
    )

    best = ranking.best()

    assert best.spacegroup_number == 12
    assert best.symprec == 0.1


def test_ranking_best_without_symmetry_raises():
    ranking = SymmetryRanking([SymmetryLevel(symprec=0.0, error='nope')])

    with pytest.raises(ValueError, match='Could not determine symmetry'):
        ranking.best()


def test_empty_ranking():
    ranking = SymmetryRanking([])

    assert len(ranking) == 0
    assert ranking.found == []
    assert ranking.candidates == []
    assert repr(ranking) == 'SymmetryRanking()'
    # just the header and its underline
    assert len(ranking.format().splitlines()) == 2

    with pytest.raises(ValueError, match='Could not determine symmetry'):
        ranking.best()


def test_ranking_format():
    ranking = SymmetryRanking(
        [
            _level(
                12, 'C2/m', symprec=0.1, deviation=0.0715, angle_deviation=0.7, n_observed=4
            ),
            SymmetryLevel(symprec=0.3, error='nope'),
        ]
    )

    table = ranking.format()
    lines = table.splitlines()

    for header in (
        'symprec (Å)',
        'deviation (Å)',
        'angle dev (°)',
        'space group',
        '# ops',
        '# orbits',
        '# hits',
    ):
        assert header in lines[0]
    # header + separator + one row per level
    assert len(lines) == 4
    assert lines[2].startswith('0.1')
    assert 'C2/m' in lines[2]
    assert '0.0715' in lines[2]
    # a level whose search failed is shown as such, with empty columns
    assert 'failed' in lines[3]
