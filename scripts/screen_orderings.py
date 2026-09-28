"""Screen orderings of a mobile species in any partially-occupied CIF.

Two methods, each a script of its own that this one wraps:

    goac      Coulomb (Ewald) sum over all ions with GOAC, sweeping the composition
              (`goac_sweep.py`)
    distance  maximise the distance between the mobile ions, no GOAC needed
              (`randomized_distribution.py`)

From the command line, everything after the method goes to that method's options:

    python scripts/screen_orderings.py goac my.cif --specie Na --supercell 2 2 2
    python scripts/screen_orderings.py distance my.cif --specie Na --workdir out/
    python scripts/screen_orderings.py goac --help

From Python, `screen` takes the same options as keyword arguments:

    from screen_orderings import screen

    result = screen('goac', 'my.cif', specie='Na', supercell=[2, 2, 2])
    keepers = screen('distance', 'my.cif', specie='Na', n_min=4, n_max=4)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import goac_sweep
import randomized_distribution

METHODS = ('goac', 'distance')


def screen(method: str, cif: str | Path, **options):
    """Run `method` on `cif`, returning a `goac_sweep.GOACResult` for 'goac'
    and the kept configurations per count for 'distance'."""
    if method == 'goac':
        return goac_sweep.GOACSweep(cif=Path(cif), **options).run()
    if method == 'distance':
        return randomized_distribution.run(Path(cif), **options)
    raise ValueError(f'method must be one of {METHODS}, not {method!r}')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    methods = parser.add_subparsers(dest='method', required=True)
    for method, module in zip(METHODS, (goac_sweep, randomized_distribution)):
        # The method's own parser supplies the options, and its -h.
        own = module.build_parser()
        methods.add_parser(
            method,
            parents=[own],
            add_help=False,
            help=own.description.splitlines()[0],
            description=own.description,
            formatter_class=own.formatter_class,
        )
    return parser


def main() -> None:
    options = vars(build_parser().parse_args())
    screen(options.pop('method'), options.pop('cif'), **options)


if __name__ == '__main__':
    main()
