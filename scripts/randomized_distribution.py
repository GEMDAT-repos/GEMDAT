"""Screen Li orderings by maximizing the Li-Li interatomic distance.

Adapted from a script by A. Klavrinenko (after T. Schwietert and N. de Klerk) so that it
runs from the command line on any platform and reads GEMDAT's partially-occupied CIFs
directly. The algorithm is unchanged:

    1. Fill every interstitial position of the chosen specie.
    2. Draw a random subset of `n` of those positions and delete the rest.
    3. Score the draw by the sum of all pairwise `specie`-`specie` distances
       (minimum-image, from pymatgen's `distance_matrix`).
    4. Keep the configurations with the highest, middle and lowest score.

Configurations with the largest total distance tend to have the lowest electrostatic
energy, considering only interactions within the specie of interest, so this is a cheap
stand-in for the Coulomb pre-screening in `goac_li3ycl3br3.py` (which scores the same
kind of random draws with a real Ewald sum over *all* ions). Use this one when you want
candidates without installing GOAC, or to sanity-check its output; prefer GOAC when the
counter-ion sublattice matters.

The input CIF may be partially occupied: `--fill` (the default) rebuilds it into the
ordered "every position taken" cell the algorithm expects. For Li3YCl3Br3-c2m.cif that
means 16 Li interstitials, Y on the two 2a positions, and 12 halide positions.

    NOTE on the halides: Cl and Br share the same crystallographic positions at 50/50
    occupancy, so an ordered cell has to commit to one arrangement. `--fill` picks a
    random half-and-half split (governed by `--seed`) and keeps it fixed for the whole
    run. This script does not optimize it — only the Li sublattice is screened. Settle
    the halide ordering separately, e.g. with DFT.

Examples
--------
    # sweep every Li count from 0 to 16 in the unit cell
    python scripts/randomized_distribution.py --workdir out/

    # a single concentration, more draws, in a 2x1x2 supercell
    python scripts/randomized_distribution.py --n-min 24 --n-max 24 \
        --supercell 2 1 2 --tries 100000 --workdir out/
"""

from __future__ import annotations

import argparse
import random
import warnings
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Poscar
from tqdm import tqdm

from gemdat.utils import DATA

DEFAULT_CIF = Path(str(DATA / 'Li3YCl3Br3-c2m.cif'))


def fill_sites(base: Structure, symbol: str, rng: random.Random) -> Structure:
    """Rebuild a partially-occupied cell with every `symbol` position taken.

    Sites that hold `symbol` (alone or shared with another element, as
    Li2/Y9 are here) become fully-occupied `symbol` sites. Any other
    disordered site is committed to a random arrangement in proportion
    to its refined occupancies, which keeps the overall stoichiometry
    while making the cell writable as a POSCAR.
    """
    species: list[str] = []
    coords: list = []

    shared: dict[tuple, list[int]] = {}
    for site in base:
        amounts = site.species.get_el_amt_dict()
        if symbol in amounts:
            species.append(symbol)
            coords.append(site.frac_coords)
        elif len(amounts) == 1:
            species.append(next(iter(amounts)))
            coords.append(site.frac_coords)
        else:
            # Disordered non-`symbol` site (the Cl/Br positions): defer, so that all
            # positions sharing the same composition are split as one group.
            key = tuple(sorted((el, round(amt, 4)) for el, amt in amounts.items()))
            shared.setdefault(key, []).append(len(species))
            species.append('')  # placeholder, filled in below
            coords.append(site.frac_coords)

    for key, indices in shared.items():
        elements = [el for el, _ in key]
        occupancies = np.array([amt for _, amt in key], dtype=float)
        counts = np.round(occupancies / occupancies.sum() * len(indices)).astype(int)
        counts[-1] = len(indices) - counts[:-1].sum()
        assignment = [el for el, count in zip(elements, counts) for _ in range(count)]
        rng.shuffle(assignment)
        for index, element in zip(indices, assignment):
            species[index] = element

    return Structure(base.lattice, species, coords)


def remove_specie(structure: Structure, n_keep: int, symbol: str) -> Structure:
    """Return a copy of `structure` with all but `n_keep` random `symbol` sites
    deleted."""
    indices = structure.indices_from_symbol(symbol)
    drop = random.sample(list(indices), len(indices) - n_keep)
    new_structure = structure.copy()
    new_structure.remove_sites(drop)
    return new_structure


def total_distance(structure: Structure, symbol: str, cutoff: float | None) -> float:
    """Sum of all pairwise `symbol`-`symbol` distances.

    Raises `ValueError` when any pair sits closer than `cutoff`, which
    the caller uses to reject the draw and try again.
    """
    indices = list(structure.indices_from_symbol(symbol))
    distances = structure.distance_matrix[indices, :][:, indices]
    pairs = distances[np.triu_indices_from(distances, k=1)]
    if cutoff is not None and np.any(pairs <= cutoff):
        raise ValueError('distances below cutoff found')
    return float(pairs.sum())


class Keeper:
    """Keeps the `size` configurations whose score is most extreme in one
    direction."""

    def __init__(self, size: int, *, best: str):
        self.size = size
        self.best = best  # 'high', 'low' or 'mid'
        self.scores: list[float] = []
        self.structures: list[Structure] = []

    def _key(self, score: float, target: float) -> float:
        if self.best == 'high':
            return -score
        if self.best == 'low':
            return score
        return abs(score - target)

    def offer(self, score: float, structure: Structure, target: float = 0.0) -> None:
        keys = [self._key(s, target) for s in self.scores]
        if len(self.scores) < self.size:
            self.scores.append(score)
            self.structures.append(structure.copy())
        elif self._key(score, target) < max(keys):
            worst = int(np.argmax(keys))
            self.scores[worst] = score
            self.structures[worst] = structure.copy()

    def sorted_pairs(self) -> list[tuple[float, Structure]]:
        """Configurations, best first."""
        order = np.argsort(self.scores)
        if self.best == 'high':
            order = order[::-1]
        return [(self.scores[i], self.structures[i]) for i in order]


def optimize(
    structure: Structure,
    *,
    n_keep: int,
    tries: int,
    symbol: str,
    cutoff: float | None,
    n_high: int,
    n_low: int,
    n_mid: int,
) -> dict[str, Keeper]:
    """Draw `tries` random configurations of `n_keep` atoms and keep the
    extremes."""
    keepers = {
        'High': Keeper(n_high, best='high'),
        'Low': Keeper(n_low, best='low'),
        'Mid': Keeper(n_mid, best='mid'),
    }

    n_sites = len(structure.indices_from_symbol(symbol))
    if n_keep in (0, n_sites):
        # Only one configuration exists; drawing more of them is pointless.
        tries = 1

    for _ in tqdm(range(tries), desc=f'{n_keep:>3d} {symbol}', unit=' tries', leave=False):
        while True:
            candidate = remove_specie(structure, n_keep, symbol)
            try:
                score = total_distance(candidate, symbol, cutoff)
            except ValueError:
                continue  # below cutoff: redraw
            break

        keepers['High'].offer(score, candidate)
        keepers['Low'].offer(score, candidate)

        # The middle is defined against the extremes found so far, so early draws are
        # judged against a moving target. This mirrors the original script.
        if keepers['High'].scores and keepers['Low'].scores:
            target = (max(keepers['High'].scores) + min(keepers['Low'].scores)) / 2
            keepers['Mid'].offer(score, candidate, target)

    return keepers


def write_results(keepers: dict[str, Keeper], workdir: Path, n_keep: int, symbol: str) -> None:
    """Write each kept configuration as a POSCAR under
    `workdir/<n>_<symbol>/<label>/`."""
    root = workdir / f'{n_keep}_{symbol}'
    for label, keeper in keepers.items():
        directory = root / label
        directory.mkdir(parents=True, exist_ok=True)
        for i, (score, structure) in enumerate(keeper.sorted_pairs(), start=1):
            path = directory / f'POSCAR_{i}'
            # Sorted, so that each element gets a single block in the POSCAR header
            # instead of one per site group.
            Poscar(structure=structure.get_sorted_structure()).write_file(str(path))
            (directory / f'POSCAR_{i}.distance').write_text(f'{score:.6f}\n')


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--cif', type=Path, default=DEFAULT_CIF, help='input structure')
    parser.add_argument('--specie', default='Li', help='specie to distribute')
    parser.add_argument(
        '--supercell',
        type=int,
        nargs=3,
        metavar=('A', 'B', 'C'),
        default=[1, 1, 1],
        help='supercell to build before screening',
    )
    parser.add_argument(
        '--n-min', type=int, default=0, help='lowest number of atoms in the output cell'
    )
    parser.add_argument(
        '--n-max',
        type=int,
        help='highest number of atoms in the output cell (default: every position)',
    )
    parser.add_argument(
        '--tries', type=int, default=1000, help='random draws per concentration'
    )
    parser.add_argument('--n-high', type=int, default=8, help='configurations to keep, highest')
    parser.add_argument('--n-low', type=int, default=4, help='configurations to keep, lowest')
    parser.add_argument('--n-mid', type=int, default=4, help='configurations to keep, middle')
    parser.add_argument(
        '--cutoff',
        type=float,
        help='reject draws with any pair closer than this, in angstrom',
    )
    parser.add_argument(
        '--no-fill',
        dest='fill',
        action='store_false',
        help='input is already ordered with every position taken',
    )
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--workdir', type=Path, default=Path('randomized_distribution'), help='output directory'
    )
    args = parser.parse_args()

    random.seed(args.seed)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # partial occupancies are expected here
        base = Structure.from_file(args.cif)

    structure = fill_sites(base, args.specie, random.Random(args.seed)) if args.fill else base
    structure.make_supercell(args.supercell)
    if not structure.is_valid():
        raise SystemExit(
            'structure is not valid: it contains atoms that are too close together'
        )

    n_sites = len(structure.indices_from_symbol(args.specie))
    n_max = n_sites if args.n_max is None else args.n_max
    if not 0 <= args.n_min <= n_max <= n_sites:
        raise SystemExit(
            f'--n-min/--n-max must satisfy 0 <= n-min <= n-max <= {n_sites} '
            f'({args.specie} positions in this cell)'
        )

    print(f'{structure.composition.reduced_formula}, {len(structure)} sites')
    print(f'{n_sites} {args.specie} positions, screening {args.n_min}..{n_max}')
    print(f'{args.tries} random draws per concentration\n')

    for n_keep in range(args.n_min, n_max + 1):
        keepers = optimize(
            structure,
            n_keep=n_keep,
            tries=args.tries,
            symbol=args.specie,
            cutoff=args.cutoff,
            n_high=args.n_high,
            n_low=args.n_low,
            n_mid=args.n_mid,
        )
        write_results(keepers, args.workdir, n_keep, args.specie)

        high = keepers['High'].sorted_pairs()
        low = keepers['Low'].sorted_pairs()
        best = high[0][0] if high else 0.0
        worst = low[0][0] if low else 0.0
        print(f'{n_keep:>3d} {args.specie}: max total distance {best:10.3f}, min {worst:10.3f}')

    print(f'\nWrote configurations to {args.workdir}')


if __name__ == '__main__':
    main()
