from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

from gemdat.symmetry import SymmetryAnalyzer, SymmetryLevel, SymmetryRanking

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
    level = SymmetryAnalyzer(ideal_structure).level(0.1)

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
    level = SymmetryAnalyzer(ideal_structure).level(0.0)

    assert level.symprec == 0.0
    assert level.spacegroup_number is None
    assert level.error


def test_level_with_deviation(ideal_structure):
    level = SymmetryAnalyzer(ideal_structure).level(0.1, with_deviation=True)

    # an undistorted structure satisfies its own symmetry exactly
    assert level.deviation == pytest.approx(0.0, abs=1e-9)
    assert level.angle_deviation == pytest.approx(0.0, abs=1e-9)


def test_scan(noisy_structure):
    ranking = SymmetryAnalyzer(noisy_structure).scan(symprec_min=0.01, symprec_max=0.5)

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
    highest = analyzer.scan(symprec_min=0.01, symprec_max=0.5)[-1]
    assert highest.spacegroup_number is not None and highest.spacegroup_number > 1

    for symprec in (highest.symprec, 0.3, 0.5):
        level = analyzer.level(symprec, with_deviation=True)
        if level.spacegroup_number != highest.spacegroup_number:
            continue
        assert level.deviation == pytest.approx(highest.deviation)
        assert level.angle_deviation == pytest.approx(highest.angle_deviation)


def test_scan_rejects_invalid_range(ideal_structure):
    analyzer = SymmetryAnalyzer(ideal_structure)

    with pytest.raises(ValueError, match='`symprec_min` must be positive'):
        analyzer.scan(symprec_min=0.0)

    with pytest.raises(ValueError, match='must not be smaller'):
        analyzer.scan(symprec_min=0.5, symprec_max=0.1)

    with pytest.raises(ValueError, match='`n_samples` must be at least 2'):
        analyzer.scan(n_samples=1)


def test_levels_skips_deviation(noisy_structure):
    ranking = SymmetryAnalyzer(noisy_structure).levels(SYMPREC_RANGE)

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

    for header in ('symprec (Å)', 'deviation (Å)', 'angle dev (°)', 'space group', '# orbits'):
        assert header in lines[0]
    # header + separator + one row per level
    assert len(lines) == 4
    assert lines[2].startswith('0.1')
    assert 'C2/m' in lines[2]
    assert '0.0715' in lines[2]
    # a level whose search failed is shown as such, with empty columns
    assert 'failed' in lines[3]
