"""Screen orderings of a mobile species in any partially-occupied CIF.

Two methods, each wrapping a script of its own:

    goac      Coulomb (Ewald) sum over all ions with GOAC (`goac_sweep.py`)
    distance  maximise the distance between the mobile ions, no GOAC needed
              (`randomized_distribution.py`)

Both take one set of options, below, which this script checks and translates to the
names and units the underlying script uses. An option that does not apply to the chosen
method is an error, on the command line and from Python alike.

Shared options
--------------
    option           meaning                                   goac          distance
    cif              input CIF, may be partially occupied      cif           cif
    specie           mobile species to order (Li)              specie        specie
    supercell        cell the orderings are drawn in (1 1 1)   supercell     supercell
    formula_units    formula units per unit cell that x is     formula_units (converts x)
                     counted per (from the CIF's host formula)
    x_min, x_max     range of x, mobile ions per formula unit  x_min, x_max  n_min, n_max
                     (0 .. every mobile position filled)
    x_step           spacing in x (one ion per unit cell)      x_step        n_step
    samples          random orderings drawn per composition    samples       tries
                     (goac 100000, distance 1000)
    n_best           lowest-energy orderings written per       n_best        n_high
                     composition (goac 1, distance 8); for
                     distance, the largest total distance
    output           directory to write to (a temp dir)        output        workdir
    quiet            don't report progress                     quiet         quiet
    progress         progress bar on stderr                    progress      progress

`distance` counts ions in the whole cell, so x is converted as n = x * formula_units *
cells, and x_min/x_step must come out as whole numbers of ions (x_max is rounded down).

goac only
---------
    charges            formal charges, {'Li': 1, ...} or Li=1 ... (guessed from the CIF)
    solver             random (as in the paper), sa, mc or ga (sa)
    steps              optimizer steps (200000)
    starts             random starting points for sa/mc/ga (20)
    tol                eV within which orderings count as the same (1e-3)
    disorder_partners  also permute elements sharing the mobile sites
    extra_interstitials, void_radius (2.3), void_separation (2.2)
                       add the empty voids of the anion packing as mobile positions
    scatter_max        sampled orderings scattered per composition in sweep.png (2000)

distance only
-------------
    n_low, n_mid       orderings kept with the lowest / middle total distance (4, 4)
    cutoff             reject draws with a pair closer than this, in Å (none)
    fill               rebuild the CIF with every position taken first (True;
                       --no-fill on the command line)
    seed               random seed (0)

Usage
-----
    python scripts/calculate_coarse_potentials.py goac my.cif --specie Na --supercell 2 2 2
    python scripts/calculate_coarse_potentials.py distance my.cif --specie Na --x-min 2
    python scripts/calculate_coarse_potentials.py goac --help

or from Python, with the same names as keyword arguments:

    from calculate_coarse_potentials import screen

    result = screen('goac', 'my.cif', specie='Na', supercell=[2, 2, 2])
    keepers = screen('distance', 'my.cif', specie='Na', x_min=2, x_max=2)
"""

from __future__ import annotations

import argparse
import math
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import goac_sweep
import numpy as np
import randomized_distribution
from pymatgen.core import Structure


def goac(
    cif: str | Path,
    *,
    specie: str = 'Li',
    supercell: list[int] | None = None,
    formula_units: int | None = None,
    x_min: float = 0.0,
    x_max: float | None = None,
    x_step: float | None = None,
    samples: int = 100000,
    n_best: int = 1,
    output: str | Path | None = None,
    quiet: bool = False,
    progress: bool = False,
    charges: dict[str, float] | list[tuple[str, float]] | None = None,
    solver: str = 'sa',
    steps: int = 200000,
    starts: int = 20,
    tol: float = goac_sweep.DEFAULT_TOL,
    disorder_partners: bool = False,
    extra_interstitials: bool = False,
    void_radius: float = 2.3,
    void_separation: float = 2.2,
    scatter_max: int = 2000,
) -> goac_sweep.GOACResult:
    """Sweep x with GOAC; see the module docstring for the options."""
    if solver not in goac_sweep.SOLVERS:
        raise ValueError(f'solver must be one of {tuple(goac_sweep.SOLVERS)}, not {solver!r}')
    return goac_sweep.GOACSweep(
        cif=Path(cif),
        specie=specie,
        supercell=supercell,
        formula_units=formula_units,
        x_min=x_min,
        x_max=x_max,
        x_step=x_step,
        samples=samples,
        n_best=n_best,
        output=Path(output) if output is not None else None,
        quiet=quiet,
        progress=progress,
        charges=charges,
        solver=solver,
        steps=steps,
        starts=starts,
        tol=tol,
        disorder_partners=disorder_partners,
        extra_interstitials=extra_interstitials,
        void_radius=void_radius,
        void_separation=void_separation,
        scatter_max=scatter_max,
    ).run()


def distance(
    cif: str | Path,
    *,
    specie: str = 'Li',
    supercell: list[int] | None = None,
    formula_units: int | None = None,
    x_min: float = 0.0,
    x_max: float | None = None,
    x_step: float | None = None,
    samples: int = 1000,
    n_best: int = 8,
    output: str | Path | None = None,
    quiet: bool = False,
    progress: bool = False,
    n_low: int = 4,
    n_mid: int = 4,
    cutoff: float | None = None,
    fill: bool = True,
    seed: int = 0,
) -> dict[int, dict[str, randomized_distribution.Keeper]]:
    """Screen by total distance; see the module docstring for the options.

    Returns the kept configurations per number of `specie` ions in the
    cell.
    """
    supercell = list(supercell or [1, 1, 1])
    if formula_units is None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # partial occupancies are expected here
            base = Structure.from_file(cif)
        formula_units = goac_sweep.count_formula_units(base, specie)
    cells = int(np.prod(supercell))
    per_x = formula_units * cells  # ions in the cell per unit of x

    def ions(x: float, name: str) -> int:
        n = x * per_x
        if abs(n - round(n)) > 1e-6:
            raise ValueError(
                f'{name}={x:g} is {n:g} {specie} ions in the cell, not a whole number '
                f'({formula_units} formula units x {cells} cells)'
            )
        return round(n)

    n_step = cells if x_step is None else ions(x_step, 'x_step')
    if n_step < 1:
        raise ValueError(f'x_step={x_step:g} is less than one {specie} ion in the cell')

    return randomized_distribution.run(
        Path(cif),
        specie=specie,
        supercell=supercell,
        n_min=ions(x_min, 'x_min'),
        n_max=None if x_max is None else math.floor(x_max * per_x + 1e-6),
        n_step=n_step,
        tries=samples,
        n_high=n_best,
        workdir=Path(output) if output is not None else None,
        quiet=quiet,
        progress=progress,
        n_low=n_low,
        n_mid=n_mid,
        cutoff=cutoff,
        fill=fill,
        seed=seed,
    )


METHODS: dict[str, Callable[..., Any]] = {'goac': goac, 'distance': distance}


def screen(method: str, cif: str | Path, **options):
    """Run `method` ('goac' or 'distance') on `cif` with `options`.

    Options that `method` does not take raise a TypeError.
    """
    if method not in METHODS:
        raise ValueError(f'method must be one of {tuple(METHODS)}, not {method!r}')
    return METHODS[method](cif, **options)


def build_parser() -> argparse.ArgumentParser:
    # Unset options are left out, so the defaults are the functions' and differ per method
    # where the table in the module docstring says so.
    shared = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS)
    group = shared.add_argument_group('shared options')
    group.add_argument('cif', type=Path, help='input CIF, may be partially occupied')
    group.add_argument('--specie', help='mobile species to order (Li)')
    group.add_argument(
        '--supercell',
        type=int,
        nargs=3,
        metavar=('NA', 'NB', 'NC'),
        help='cell the orderings are drawn in (1 1 1)',
    )
    group.add_argument(
        '--formula-units',
        type=int,
        help='formula units per unit cell that x is counted per (from the CIF)',
    )
    group.add_argument('--x-min', type=float, help='lowest x, mobile ions per f.u. (0)')
    group.add_argument('--x-max', type=float, help='highest x (every position filled)')
    group.add_argument('--x-step', type=float, help='spacing in x (one ion per unit cell)')
    group.add_argument(
        '--samples',
        type=int,
        help='random orderings per composition (goac 100000, distance 1000)',
    )
    group.add_argument(
        '--n-best',
        type=int,
        help='lowest-energy orderings written per composition (goac 1, distance 8)',
    )
    group.add_argument('--output', type=Path, help='output directory (a temp dir)')
    group.add_argument('--quiet', action='store_true', help="don't report progress")
    group.add_argument('--progress', action='store_true', help='progress bar on stderr')

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    methods = parser.add_subparsers(dest='method', required=True)

    own = methods.add_parser(
        'goac', parents=[shared], argument_default=argparse.SUPPRESS, help='GOAC Coulomb sweep'
    ).add_argument_group('goac options')
    own.add_argument(
        '--charges',
        type=goac_sweep.parse_charges,
        nargs='+',
        metavar='EL=Q',
        help='formal charges, e.g. Li=1 Y=3 Cl=-1 Br=-1 (guessed from the CIF)',
    )
    own.add_argument('--solver', choices=tuple(goac_sweep.SOLVERS), help='GOAC solver (sa)')
    own.add_argument('--steps', type=int, help='optimizer steps (200000)')
    own.add_argument('--starts', type=int, help='starting points for sa/mc/ga (20)')
    own.add_argument('--tol', type=float, help='eV within which orderings are equal (1e-3)')
    own.add_argument(
        '--disorder-partners',
        action='store_true',
        help='also permute the elements sharing the mobile sites',
    )
    own.add_argument(
        '--extra-interstitials',
        action='store_true',
        help='add the empty voids of the anion packing as mobile positions',
    )
    own.add_argument('--void-radius', type=float, help='min. Å from an anion to a void (2.3)')
    own.add_argument(
        '--void-separation', type=float, help='min. Å from a void to other cations (2.2)'
    )
    own.add_argument('--scatter-max', type=int, help='orderings scattered in sweep.png (2000)')

    own = methods.add_parser(
        'distance',
        parents=[shared],
        argument_default=argparse.SUPPRESS,
        help='maximise the distance between the mobile ions',
    ).add_argument_group('distance options')
    own.add_argument('--n-low', type=int, help='orderings kept, lowest total distance (4)')
    own.add_argument('--n-mid', type=int, help='orderings kept, middle total distance (4)')
    own.add_argument('--cutoff', type=float, help='reject pairs closer than this, in Å')
    own.add_argument(
        '--no-fill',
        dest='fill',
        action='store_false',
        help='input is already ordered with every position taken',
    )
    own.add_argument('--seed', type=int, help='random seed (0)')
    return parser


def main() -> None:
    options = vars(build_parser().parse_args())
    screen(options.pop('method'), options.pop('cif'), **options)


if __name__ == '__main__':
    main()
