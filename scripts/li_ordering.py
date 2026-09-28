"""Screen Li orderings in Li3YCl3Br3, by one of two methods.

goac      Coulomb (Ewald) screening over all ions with GOAC
(`goac_li3ycl3br3.py`) distance  maximise the Li-Li distance, no GOAC
needed (`randomized_distribution.py`)

Everything after the method goes to that script's own parser:

python scripts/li_ordering.py goac --supercell 2 1 2 --samples 100000
python scripts/li_ordering.py distance --n-min 24 --n-max 24 --workdir
out/ python scripts/li_ordering.py distance --help
"""

from __future__ import annotations

import importlib
import sys

METHODS = {'goac': 'goac_li3ycl3br3', 'distance': 'randomized_distribution'}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in METHODS:
        raise SystemExit(__doc__)
    method = sys.argv.pop(1)
    sys.argv[0] = f'{sys.argv[0]} {method}'
    # Imported lazily so `distance` runs without GOAC installed.
    importlib.import_module(METHODS[method]).main()


if __name__ == '__main__':
    main()
