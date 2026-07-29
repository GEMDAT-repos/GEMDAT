"""Build a formation-energy convex hull from ordering candidates with a
universal MLIP.

This is the cheap stand-in for the DFT step behind Fig. 2a of

    Beneficial redox activity of halide solid electrolytes empowering high-performance
    anodes in all-solid-state batteries, https://doi.org/10.26434/chemrxiv-2024-x2rld

`goac_li3ycl3br3.py` (or `randomized_distribution.py`) generates low-energy Li orderings
per composition; the paper relaxes those with DFT and takes the convex hull. A universal
machine-learning interatomic potential (MACE-MP, CHGNet, ...) relaxes the same candidates
in seconds instead of core-hours, so you can see whether the hull even has the right
topology before committing any DFT. Validate the hull *vertices* with real DFT afterwards
-- MLIPs are trained on Materials-Project-style data and can be shaky on exactly the
Y3+ -> Y reduction that compensates lithiation here.

Two caveats specific to this system:

  Halide identity. `goac_li3ycl3br3.py` collapses Cl/Br onto one species (a point-charge
  model cannot tell them apart), so its output CIFs are all-Cl -- the wrong compound for a
  real energy. Unlike point charges, an MLIP *can* distinguish the halogens, so this script
  restores a 3:3 Cl:Br split. It is only the *Li* ordering that GOAC optimized; the halide
  arrangement is unsettled, so `--halide-samples k` tries k random splits per candidate and
  keeps the lowest energy. Inputs that already carry Br (e.g. randomized_distribution
  POSCARs) are left alone unless `--reassign-halides` is given.

  Reference states. As in Fig. 2a, energies are referenced to a straight line between the
  lowest- and highest-x candidates present, so a stable composition dips below the line.
  Include x=0 and the fully lithiated end in the sweep to match the paper's endpoints.

Reduction-aware mode (`--reduction-aware`)
------------------------------------------
The whole point of this composition axis is a *redox* process: each added neutral Li
donates one electron, and (charge neutrality of Li_xYCl3Br3 fixing Y's oxidation state at
`6 - x`) those electrons reduce Y3+ -> Y0 by dropping into Y-4d states as `x` climbs from
the nominal x=3 compound toward x=6. A plain MACE-MP model sees only positions and species
-- no electrons, no oxidation state -- which is exactly why its Y3+ -> Y reduction energy is
shaky (see the caveat above and `mlip_hull_reduction_notes.md`).

`--reduction-aware` computes that redox state from the composition and hands it to the
potential the way a charge/spin-conditioned MLIP wants it: `atoms.info['charge']` and
`atoms.info['spin']` (the keys MACE's global embeddings read), plus per-atom initial
magnetic moments that put the `x - 3` reducing electrons per Y (capped at the 3 that empty
Y3+ -> Y0) on the Y sublattice. The cell stays neutral; the redox information lives in the
spin. `--spin-convention` picks what goes in `atoms.info['spin']`: `multiplicity` (2S+1,
MACE's convention, the default) or `moment` (the bare unpaired-electron count 2S).

There is no pretrained checkpoint worth pointing `--model-path` at. The charge/spin-
conditioned foundation models that exist -- MACE-OMOL, MACE-POLAR-1, Meta's UMA -- are all
conditioned for *molecules*: UMA only uses charge/spin on its `omol` task and ignores them
for periodic inorganic crystals, and the MACE ones are trained on isolated molecules at a
molecular level of theory, nowhere near an ionic halide lattice. So `--model-path` expects a
checkpoint *you* fine-tuned on DFT data for this system (MACE >= 0.3.14, `--embedding_specs`
plus `--use_embedding_readout`); `mlip_hull_reduction_notes.md` sketches how.

Note also what conditioning can and cannot buy here: the cell is neutral by construction, so
the charge channel is a constant, and the spin is a deterministic function of `x`, which the
model already sees through the composition. The conditioning only changes an energy if the
training data contains the Y3+ -> Y0 reduction path itself. With a stock foundation model it
is recorded per structure and seeds a physically sensible spin state for the relaxation, but
the model ignores it for the energy (it is positions-only) -- so validate reduced vertices
with DFT regardless.

MACE is the default and needs `pip install mace-torch`; CHGNet needs `pip install chgnet`.

Examples
--------
    # relax everything a sweep wrote, build the hull
    python scripts/goac_li3ycl3br3.py sweep --n-best 8 --workdir out/
    python scripts/mlip_hull.py --workdir out/

    # or point straight at structure files, try 4 halide arrangements each, relax cells
    python scripts/mlip_hull.py 'out/*.cif' --halide-samples 4 --relax-cell

    # reduction-aware hull with your own charge/spin-conditioned MACE checkpoint
    python scripts/mlip_hull.py --workdir out/ --reduction-aware --model-path yclbr_qspin.model

Everything a run produces goes into `--output`: the relaxed geometries -- the ones the
energies actually refer to, so the ones worth looking at -- next to the hull itself
(`hull.png`), a `hull.csv` saying which of them are vertices of it, and a `hull.json` with
what neither of those records. That is a fresh temp directory unless one is named:

    python scripts/mlip_hull.py --workdir out/ --output relaxed/

Programmatic use mirrors the CLI, split in two, the same way `goac_li3ycl3br3.py` is: a
`HullBuilder` holds the settings and does the relaxing, and hands back a `HullResult`
holding what it relaxed, which does the reporting on it. The builder's constructor takes the
same options as the argument parser (`HullBuilder(**vars(args))` is exactly what `main`
does), so

    HullBuilder(inputs=['out/*.cif'], halide_samples=4, relax_cell=True).run()

is the in-process equivalent of the second example above. `run` reports as it goes unless
`quiet=True` (`--quiet`), and `progress=True` (`--progress`) puts a bar over the candidates
on stderr; the two combine, for a bar and nothing else. Relaxing is by far the expensive
part, and the result keeps every relaxed structure with its energy, so the hull can be
tabulated (`summary`), plotted (`plot`, `save_figure`) and written out (`save_csv`,
`save_relaxed`) again without paying for the MLIP twice:

    hull = HullBuilder(workdir=Path('out/'), quiet=True).run()
    hull.model = 'MACE-MP medium'
    hull.save_figure(Path('elsewhere.png'))

Those three files are also all the reporting half needs, so it outlives the process that
did the relaxing: a `HullResult` handed nothing but a finished output directory reads the
relaxations back out of it and picks the run up where it left off, days later and without
the potential (or the inputs) anywhere in sight:

    hull = HullResult(workdir=Path('relaxed/'))
    print(hull.summary())
    hull.restrict((3, 8)).save_figure(Path('hull_3to8.png'))

The energies come from `hull.csv`, the geometries from the relaxed structures its `relaxed`
column points at (by basename next to the csv, if the directory has since moved), and
everything else -- which model, which reference line, which convergence settings -- from
`hull.json`. Only what the save format carries survives, so a reduction-aware run's magnetic
moments need `--save-format extxyz` to come back; the hull itself does not depend on them.
Anything passed alongside `workdir` wins over what is stored there, so
`HullResult(workdir=..., model='MACE-MP medium')` restores with a retitled figure.

The hull a full sweep gives is the one against its two extreme compositions, which is not the
only interesting one: what happens as the nominal x=3 compound takes up lithium is a hull
referenced to x=3, not to the delithiated end. `restrict` cuts the relaxations to a
composition window and hands back a `HullResult` over what is left, reporting on itself the
same way -- including a hull recomputed over the remaining candidates, which need not have
the same vertices:

    window = hull.restrict((3, 8))          # both ends at 0 by construction
    print(window.summary())
    window.save_figure(Path('hull_3to8.png'))

    hull.restrict((3, 8), reference='parent')   # keeps the full sweep's zero instead

`plot`/`save_figure` take `xlim=` (and `reference=`) directly for the figure alone.

The individual steps (`discover_inputs`, `load_calculator`, `halide_variants`,
`relax_candidate`, ...) are methods on the builder, reading their configuration off it. One
option has no command-line counterpart: `calculator=` takes a ready-made ASE calculator and
bypasses `--model`, so any potential ASE can drive works, not just the two the CLI knows.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
import tempfile
import warnings
from dataclasses import dataclass, field, fields
from functools import cached_property
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from ase.optimize import FIRE
from matplotlib.figure import Figure
from tqdm import tqdm

try:  # ASE >= 3.23
    from ase.filters import FrechetCellFilter as CellFilter
except ImportError:  # pragma: no cover - older ASE
    from ase.constraints import ExpCellFilter as CellFilter

HALIDES = ('Cl', 'Br')

# What a finished run leaves in its output directory. The first three are what a later
# `HullResult(workdir=...)` reads back: hull.csv holds the energies, its `relaxed` column
# points at the geometries they belong to, and hull.json carries the rest of the state that
# is nowhere in either (which model, which reference line, which relaxation settings).
HULL_CSV = 'hull.csv'
HULL_META = 'hull.json'
HULL_FIGURE = 'hull.png'

# Formats the relaxed structures can be written in, mapped to their file extension. extxyz
# is the only one that carries the energy and the reduction-aware moments along with the
# geometry; CIF and POSCAR are geometry-only but are what VESTA/pymatgen expect.
SAVE_FORMATS = {'cif': '.cif', 'extxyz': '.xyz', 'vasp': '.vasp'}

# What `--reduction-aware` writes into `atoms.info['spin']`. MACE's charge/spin embeddings
# (and the OMOL model built on them) take the spin *multiplicity* 2S+1; 'moment' keeps the
# bare unpaired-electron count 2S for checkpoints trained the other way round.
SPIN_CONVENTIONS = ('multiplicity', 'moment')

# Molecule-trained charge/spin embeddings cover a narrow spin window (the documented MACE
# `--embedding_specs` example runs 0-4). A supercell reduces every Y at once, so the cell
# multiplicity blows past that immediately -- worth saying out loud once.
TYPICAL_SPIN_EMBED_MAX = 4.0


class Table(str):
    """A block of text that shows itself rather than its repr.

    A notebook cell ending in `result.summary()` displays the returned
    value's *repr*, which for a plain string puts the whole table on one
    line with its newlines escaped to `\\n`. IPython's pretty printer
    asks for `_repr_pretty_` first, and `__repr__` is overridden to
    match, so the table comes out laid out from a notebook cell, from
    the plain REPL and from `print` alike.

    Deliberately duplicated from `goac_li3ycl3br3.py` rather than
    shared: importing that module raises SystemExit when GOAC is not
    installed, which is no reason for this script to stop working.
    """

    def __repr__(self) -> str:
        return str(self)

    def _repr_pretty_(self, printer, cycle) -> None:
        printer.text(str(self))


@dataclass
class Result:
    """The kept (lowest-energy) relaxation for one candidate structure."""

    x: float  # Li per YCl3Br3 formula unit
    energy_per_fu: float  # eV per formula unit
    n_fu: int
    source: Path
    converged: bool
    atoms: Atoms  # the relaxed geometry, results frozen onto a SinglePointCalculator

    @property
    def energy(self) -> float:
        """Total energy of the relaxed cell, in eV."""
        return self.energy_per_fu * self.n_fu

    @property
    def n_reducing_e(self) -> float:
        """Electrons per Y this composition puts into Y-4d (see
        `apply_reduction_state`)."""
        return float(np.clip(self.x - 3.0, 0.0, 3.0))


def normalize_device(value: str) -> str:
    """Validate a torch device string, accepting 'gpu' as an alias for 'cuda'.

    Torch has no device called 'gpu': it reaches
    `torch.load(map_location=...)` unrecognized and only blows up deep
    inside unpickling ("don't know how to restore data location ...
    tagged with gpu"), after the foundation model has already been
    downloaded. Catch it here instead.
    """
    device = value.lower()
    if device == 'gpu':
        return 'cuda'
    head = device.split(':')[0]
    if head not in ('cpu', 'cuda', 'mps', 'xpu'):
        raise argparse.ArgumentTypeError(
            f'unknown device {value!r}; use cpu, cuda (or gpu), or mps'
        )
    return device


def composition_x(atoms: Atoms) -> tuple[int, int, float]:
    """Return (n_Li, n_formula_units, x) where x = Li per YCl3Br3 formula unit.

    One Y per formula unit, so the Y count is the number of formula units. Structures
    without Y cannot be placed on the composition axis and are rejected by the caller.
    """
    symbols = atoms.get_chemical_symbols()
    n_li = symbols.count('Li')
    n_y = symbols.count('Y')
    x = n_li / n_y if n_y else float('nan')
    return n_li, n_y, x


def apply_reduction_state(
    atoms: Atoms, x: float, *, spin_convention: str = 'multiplicity'
) -> tuple[float, float, float]:
    """Encode the Li_xYCl3Br3 redox state on `atoms` for a charge/spin-
    conditioned MLIP.

    Charge neutrality of Li_xYCl3Br3 (Li+ counter-ions, closed-shell -1 halides) fixes Y's
    oxidation state at `6 - x`, so the `x - 3` added electrons per formula unit reduce Y3+
    toward Y0, capped at the 3 electrons that empty Y3+ -> Y0. Those reducing electrons enter
    Y-4d (early-d, so high-spin: one unpaired moment each), while Li is Li+ and the halides
    are closed-shell -- all nominally spinless.

    The cell stays neutral; the redox information lives in the spin. This sets per-atom
    initial magnetic moments (the reducing electrons on Y, zero elsewhere), a zero total
    charge, and the total spin on `atoms.info`, and returns `(n_reducing_e_per_Y, q_Y,
    spin)` for reporting.

    The moments sum to the unpaired-electron count 2S, but that is *not* what MACE's spin
    embedding takes: `atoms.info['spin']` there is the spin multiplicity 2S+1 (an OMOL
    singlet is `spin = 1`, not 0), so writing 2S would sit one whole unit off and, at the
    delithiated end, land on a value the embedding never sees. `spin_convention` defaults to
    that multiplicity and keeps `'moment'` for checkpoints trained on the bare count.

    Structures with x < 3 would require oxidising the halides rather than reducing Y, which
    this bookkeeping cannot represent, so the reduction is clamped to zero there.
    """
    if spin_convention not in SPIN_CONVENTIONS:  # pragma: no cover - argparse guards this
        raise ValueError(f'unknown spin convention {spin_convention!r}')

    n_reducing = float(np.clip(x - 3.0, 0.0, 3.0))  # electrons per Y entering Y-4d
    q_y = 3.0 - n_reducing  # Y oxidation state after reduction

    symbols = np.array(atoms.get_chemical_symbols())
    magmoms = np.where(symbols == 'Y', n_reducing, 0.0)
    atoms.set_initial_magnetic_moments(magmoms)

    unpaired = float(magmoms.sum())  # 2S: total unpaired moment across the cell
    spin = unpaired + 1.0 if spin_convention == 'multiplicity' else unpaired
    atoms.info['charge'] = 0
    atoms.info['spin'] = spin
    return n_reducing, q_y, spin


def lower_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Lower convex envelope of 2D points (Andrew's monotone chain, lower part
    only)."""
    # Only the lowest point per composition can be a vertex, and dropping the rest up front
    # matters at the ends: a vertical stack of candidates at the last x has nothing after it
    # to pop the higher ones back off, so the chain would otherwise finish on one of them.
    best: dict[float, float] = {}
    for x, y in points:
        if x not in best or y < best[x]:
            best[x] = y

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    hull: list[tuple[float, float]] = []
    for p in sorted(best.items()):
        while len(hull) >= 2 and cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)
    return hull


def locate_relaxed(name: str, workdir: Path) -> Path:
    """Where a `hull.csv` `relaxed` entry actually is now.

    The column holds the path as it was written, which is relative to
    the CWD of the run rather than to the workdir, so a directory that
    has since been moved -- or is simply being read from somewhere else
    -- only still matches by basename next to the csv.
    """
    if not name:
        raise SystemExit(
            'a hull.csv row has no relaxed structure to restore from; the geometries are '
            'only written by save_relaxed, which run() calls before writing the csv'
        )
    path = Path(name)
    if path.exists():
        return path
    fallback = workdir / path.name
    if fallback.exists():
        return fallback
    raise SystemExit(f'relaxed structure {name} is missing, and is not in {workdir} either')


def read_metadata(workdir: Path) -> dict:
    """The `hull.json` state of a finished run, as `HullResult` keyword
    arguments.

    Empty when there is no such file, so a workdir written before this
    existed (or by hand) still restores -- just with the defaults for
    what it cannot say.
    """
    path = workdir / HULL_META
    if not path.exists():
        return {}
    state = json.loads(path.read_text())
    if state.get('xlim') is not None:
        state['xlim'] = tuple(state['xlim'])
    if state.get('reference_ends') is not None:
        state['reference_ends'] = tuple(tuple(end) for end in state['reference_ends'])
    # A stored null is the field's own default anyway; dropping it keeps the caller's value.
    return {key: value for key, value in state.items() if value is not None}


def read_workdir(workdir: Path) -> tuple[list[Result], dict[Path, Path]]:
    """Read a finished run's relaxations back out of `workdir`.

    One `Result` per `hull.csv` row, with the relaxed geometry read from
    the file that row points at and the csv's energy frozen onto it,
    plus the source -> relaxed mapping the csv records. This is what
    `HullResult(workdir=...)` restores itself from; the energies and
    compositions come from the csv rather than being recomputed, so
    nothing here needs the MLIP (or the inputs the run was given). They
    come back to the nine decimals it is written to -- nanoelectronvolts
    on formation energies of tens of meV.

    Only what the chosen `--save-format` carries survives the round
    trip: cif and POSCAR are geometry-only, so a reduction-aware run's
    initial magnetic moments come back zeroed unless it wrote extxyz.
    Nothing the hull is read off depends on them.
    """
    path = workdir / HULL_CSV
    if not path.exists():
        raise SystemExit(
            f'nothing to restore from {workdir}: no {HULL_CSV}. A workdir can be picked up '
            'again once a run has written its relaxed structures, hull.csv and hull.json to it'
        )
    with path.open() as f:
        rows = [row for row in csv.DictReader(f) if row.get('x')]
    if not rows:
        raise SystemExit(f'{path} holds no candidates')

    results: list[Result] = []
    relaxed: dict[Path, Path] = {}
    for row in rows:
        source = Path(row['source'])
        target = locate_relaxed(row['relaxed'], workdir)
        n_fu = int(row['n_formula_units'])
        e_pfu = float(row['energy_per_fu'])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            atoms = read(str(target))
        # extxyz brings the forces back with the geometry; the energy comes from the csv
        # either way, so cif and POSCAR restore the same numbers with a bare geometry.
        forces = atoms.calc.results.get('forces') if atoms.calc is not None else None
        atoms.calc = SinglePointCalculator(atoms, energy=e_pfu * n_fu, forces=forces)
        results.append(
            Result(
                x=float(row['x']),
                energy_per_fu=e_pfu,
                n_fu=n_fu,
                source=source,
                converged=bool(int(row['converged'])),
                atoms=atoms,
            )
        )
        relaxed[source] = target
    return results, relaxed


@dataclass
class HullResult:
    """What a `HullBuilder` relaxed, and the hull read off it.

    The builder hands one of these back from `run`. It owns everything
    that reads the relaxed energies rather than produces them, so a
    finished set of relaxations can be tabulated, plotted and written
    out again, with different settings, without paying the MLIP twice.

    Constructed without any relaxations, it reads them back out of
    `workdir` instead -- the relaxed structures, `hull.csv` and
    `hull.json` a run leaves behind -- so that outliving the process
    which produced them is all it takes to report on a hull again:

        HullResult(workdir=Path('relaxed/')).summary()

    Anything else passed alongside `workdir` wins over what is stored
    there, so a restored result can still be retitled or re-referenced.
    """

    #: One kept relaxation per usable input structure, in the order they were relaxed.
    #: Left empty, they are restored from `workdir` (see `read_workdir`).
    results: list[Result] = field(default_factory=list)
    #: The MLIP that produced the energies; titles the figure.
    model: str = 'mace'
    #: Where the relaxed structures were written, where `save_figure` puts hull.png -- and
    #: what a result with no relaxations of its own restores itself from.
    workdir: Path = Path('.')
    #: Filled in by `save_relaxed`: source path -> written path, quoted in the csv.
    relaxed: dict[Path, Path] = field(default_factory=dict)
    #: The composition window this result was cut to, set by `restrict`. Only labels the
    #: figure and fixes its x range -- what was cut is simply absent from `results`.
    xlim: tuple[float, float] | None = None
    #: End members every formation energy is referenced to, as ((x, E/f.u.), (x, E/f.u.)).
    #: None (the default) uses the lowest structure at each end of the compositions present,
    #: which is what Fig. 2a does; `restrict(..., reference='parent')` fills it in with the
    #: wider set's ends so a window keeps the zero it had before being cut out of it.
    reference_ends: tuple[tuple[float, float], tuple[float, float]] | None = None
    #: The optimizer settings the relaxations ran with. They only name what produced the
    #: results in `caveats`' warning about structures that hit the step cap, but are kept
    #: (and stored) so a restored result says exactly what the run that wrote it said.
    steps: int | None = None
    fmax: float | None = None

    def __post_init__(self) -> None:
        if not self.results:
            self.restore()
        if len({result.x for result in self.results}) < 2:
            raise SystemExit('need at least two distinct compositions to reference a hull')

    def restore(self) -> None:
        """Fill this result in from a finished run's `workdir`.

        Called by the constructor when it is handed no relaxations.
        Whatever *was* passed is kept: only fields still at their
        default take the value `hull.json` recorded for them.
        """
        self.results, relaxed = read_workdir(self.workdir)
        self.relaxed = {**relaxed, **self.relaxed}
        defaults = {f.name: f.default for f in fields(self)}
        for name, value in read_metadata(self.workdir).items():
            if name in defaults and getattr(self, name) == defaults[name]:
                setattr(self, name, value)

    @property
    def xs(self) -> np.ndarray:
        """Composition of each kept structure, in relaxation order."""
        return np.array([result.x for result in self.results])

    @property
    def energies_per_fu(self) -> np.ndarray:
        """Relaxed energy per formula unit, as `GOACResult.e_min` reports it
        too."""
        return np.array([result.energy_per_fu for result in self.results])

    @property
    def converged(self) -> np.ndarray:
        """Whether each kept relaxation reached fmax before the step cap."""
        return np.array([result.converged for result in self.results])

    @cached_property
    def reference_indices(self) -> tuple[int, int]:
        """Indices of the two end members every formation energy is measured
        against.

        The lowest structure at each end of the composition range. Kept
        as indices, not just energies: if one of them failed to
        converge, every other formation energy is measured against a bad
        zero, which `caveats` says out loud.
        """
        xs, e = self.xs, self.energies_per_fu
        lo = np.flatnonzero(xs == xs.min())
        hi = np.flatnonzero(xs == xs.max())
        return int(lo[np.argmin(e[lo])]), int(hi[np.argmin(e[hi])])

    @cached_property
    def e_form(self) -> np.ndarray:
        """Formation energy per f.u.

        against the end members present, as in Fig. 2a -- or against
        `reference_ends`, if a `restrict`ed result was told to keep the
        zero of the set it was cut out of. The tie-line is the same
        function of x either way, so a window referenced to a wider
        sweep's ends reproduces that sweep's numbers exactly.
        """
        xs, e = self.xs, self.energies_per_fu
        if self.reference_ends is None:
            lo_idx, hi_idx = self.reference_indices
            ends = ((xs.min(), e[lo_idx]), (xs.max(), e[hi_idx]))
        else:
            ends = self.reference_ends
        (x_lo, e_lo), (x_hi, e_hi) = ends
        frac = (xs - x_lo) / (x_hi - x_lo)
        return e - (e_lo + (e_hi - e_lo) * frac)

    def restrict(self, xlim: tuple[float, float], *, reference: str = 'range') -> HullResult:
        """The same relaxations, cut to `xlim` and re-referenced -- a hull over
        part of the composition axis.

        The hull through a sweep says which compositions are stable
        against its two extreme ones. That is the wrong question if the
        delithiated end is not the state of interest: for what happens
        once Li_3YCl_3Br_3 starts taking up lithium, the tie-line to
        measure against runs from x=3, not from x=0. `restrict((3, 8))`
        drops everything outside the window and hands back a
        `HullResult` over what is left, which reports on itself exactly
        like the full one does -- `summary`, `plot`, `save_csv`, and a
        hull recomputed over the remaining candidates (a vertex of the
        full hull need not be one of this one, and vice versa).

        `reference` picks what the formation energies are measured
        against:

        'range'   the end members of the window, so both its ends sit at
        0 by construction           and a composition between them dips
        below the line if it is stable           against them. The
        default, and simply what any `HullResult` does with the
        compositions it holds. 'parent'  this result's own reference
        line, kept as-is, so the numbers stay           comparable with
        the wider set the window came out of and its ends do not
        generally land on 0.
        """
        if reference not in ('range', 'parent'):
            raise ValueError(f"unknown reference {reference!r}; use 'range' or 'parent'")

        lo, hi = float(min(xlim)), float(max(xlim))
        kept = [result for result in self.results if lo <= result.x <= hi]
        if not kept:
            raise SystemExit(f'no candidates with {lo:g} <= x <= {hi:g}')

        ends = None
        if reference == 'parent':
            if self.reference_ends is not None:
                ends = self.reference_ends
            else:
                lo_idx, hi_idx = self.reference_indices
                xs, e = self.xs, self.energies_per_fu
                ends = (
                    (float(xs[lo_idx]), float(e[lo_idx])),
                    (float(xs[hi_idx]), float(e[hi_idx])),
                )

        return HullResult(
            kept,
            model=self.model,
            workdir=self.workdir,
            # Shared so a csv written from the window still points at the relaxed geometries.
            relaxed=dict(self.relaxed),
            xlim=(lo, hi),
            reference_ends=ends,
            steps=self.steps,
            fmax=self.fmax,
        )

    @cached_property
    def hull(self) -> list[tuple[float, float]]:
        """The lower convex hull, as (x, E_form) vertices ordered by x."""
        return lower_hull(list(zip(self.xs.tolist(), self.e_form.tolist())))

    def is_vertex(self, x: float, e_form: float) -> bool:
        """Whether the point (x, e_form) is a vertex of `hull`."""
        vertices = {round(hx, 6): hy for hx, hy in self.hull}
        return abs(vertices.get(round(x, 6), np.inf) - e_form) < 1e-9

    @cached_property
    def on_hull(self) -> np.ndarray:
        """Boolean mask over `results`: which candidates are hull vertices."""
        return np.array(
            [self.is_vertex(x, ef) for x, ef in zip(self.xs.tolist(), self.e_form.tolist())]
        )

    def summary(self) -> Table:
        """Formation energy against composition, as a printable table."""
        lines = ['  x      E_form (eV/f.u.)   on hull   converged']
        rows = sorted(zip(self.xs.tolist(), self.e_form.tolist(), self.converged.tolist()))
        for x, ef, ok in rows:
            vertex = '*' if self.is_vertex(x, ef) else ' '
            lines.append(
                f'  {x:5.2f}  {ef:12.3f}       {vertex}         {"yes" if ok else "NO"}'
            )
        return Table('\n'.join(lines))

    def caveats(self, *, steps: int | None = None, fmax: float | None = None) -> Table:
        """Everything that makes this hull less trustworthy than it looks.

        Namely a composition range that never reaches the delithiated
        end, and relaxations that ran out of steps -- of which the ones
        that matter are the hull vertices (they set the shape) and the
        two end members (they set the reference every other energy is
        measured against). `steps` and `fmax` only name the settings
        that produced the results in the message; the finding does not
        depend on them, and they default to the ones the result was
        built (or restored) with.
        """
        steps = self.steps if steps is None else steps
        fmax = self.fmax if fmax is None else fmax
        lines = []
        x_lo = float(self.xs.min())
        # A restricted result stops at its window on purpose, so the missing delithiated end
        # is the request, not an oversight.
        if x_lo > 0 and self.xlim is None:
            lines.append(
                f'warning: lowest composition is x={x_lo:g}, not the delithiated x=0 endpoint'
            )

        n_bad = int((~self.converged).sum())
        if n_bad:
            cap = f'the {steps}-step cap' if steps is not None else 'the step cap'
            target = f'fmax={fmax} eV/A' if fmax is not None else 'fmax'
            lines.append(
                f'warning: {n_bad}/{len(self.results)} structure(s) hit {cap} '
                f'without reaching {target}; their energies are upper bounds'
            )
            bad_vertices = int((~self.converged & self.on_hull).sum())
            if bad_vertices:
                plural = 'them sit' if bad_vertices > 1 else 'them sits'
                lines.append(
                    f'         {bad_vertices} of {plural} on the hull -- '
                    'the hull shape is suspect'
                )
            # With `reference_ends` the zero comes from structures outside this result, so
            # whether it is sound is a question for the result it was taken from.
            if self.reference_ends is None and not all(
                self.converged[i] for i in self.reference_indices
            ):
                lines.append(
                    '         an end member is unconverged -- every E_form is shifted by it'
                )
            lines.append('         re-run those with a larger --steps before trusting the hull')
        return Table('\n'.join(lines))

    def plot(
        self, *, xlim: tuple[float, float] | None = None, reference: str = 'range'
    ) -> Figure:
        """The candidates, the hull through them, and what did not converge.

        As in Fig. 2a of the paper: formation energies against the end
        members present, so a stable composition dips below zero.

        `xlim` plots the hull over that composition window instead of
        the whole sweep, `reference` saying whether the window brings
        its own zero along ('range', the default) or keeps this result's
        ('parent'). Both are handed straight to `restrict`, so

        hull.plot(xlim=(3, 8))

        is shorthand for `hull.restrict((3, 8)).plot()` -- use the
        latter when the table and the csv should be cut to the window
        too.
        """
        if xlim is not None:
            return self.restrict(xlim, reference=reference).plot()

        xs, e_form, converged = self.xs, self.e_form, self.converged
        hull_x = np.array([p[0] for p in self.hull])
        hull_y = np.array([p[1] for p in self.hull])
        n_bad = int((~converged).sum())

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(
            xs[converged],
            e_form[converged],
            s=12,
            color='lightgrey',
            zorder=1,
            label='candidates',
        )
        if n_bad:
            # Drawn on top of the hull so a bad vertex is visible rather than hidden under it.
            ax.scatter(
                xs[~converged],
                e_form[~converged],
                s=28,
                marker='x',
                color='darkorange',
                zorder=3,
                label=f'unconverged ({n_bad})',
            )
        ax.plot(hull_x, hull_y, 'o-', color='crimson', zorder=2, label='convex hull')
        if self.xlim is None or self.xlim[0] <= 3.0 <= self.xlim[1]:
            ax.axvline(3.0, color='grey', linestyle=':', label='nominal Li$_3$YCl$_3$Br$_3$')
        ax.axhline(0.0, color='black', linewidth=0.6, zorder=0)
        ax.set_xlabel('x in Li$_x$YCl$_3$Br$_3$')

        # Say which zero the energies are measured against: a window carries its own unless
        # it was told to keep the wider set's, and either way it is not 'the end members'
        # without qualification once the axis has been cut.
        if self.reference_ends is not None:
            ends = 'the full set'
        elif self.xlim is not None:
            ends = f'x={xs.min():g}/{xs.max():g}'
        else:
            ends = 'end members'
        ax.set_ylabel(f'formation energy rel. to {ends} (eV/f.u.)')

        title = f'{self.model.upper()} convex hull'
        if self.xlim is not None:
            lo, hi = self.xlim
            ax.set_xlim(lo - 0.15, hi + 0.15)
            title += f', x = {lo:g}-{hi:g}'
        ax.set_title(title)
        ax.legend()

        fig.tight_layout()
        return fig

    def save_figure(
        self,
        path: Path | None = None,
        *,
        xlim: tuple[float, float] | None = None,
        reference: str = 'range',
    ) -> Path:
        """Write `plot` next to the relaxed structures as hull.png, or wherever
        `path` says.

        `xlim` and `reference` go to `plot`; name a `path` when saving a
        window as well as the full hull, or the second call overwrites
        the first.
        """
        figure = path if path is not None else self.workdir / HULL_FIGURE
        self.plot(xlim=xlim, reference=reference).savefig(figure, dpi=150)
        return figure

    def save_relaxed(self, outdir: Path | None = None, fmt: str = 'cif') -> dict[Path, Path]:
        """Write each kept relaxed structure to `outdir` (`workdir` by
        default), returning source -> written path.

        These are the geometries the energies actually refer to, which is what you want to
        look at: the Li ordering after it has moved off the ideal lattice sites GOAC placed
        it on, and the Cl/Br arrangement of the variant that won. Atoms are wrapped into the
        cell first so a viewer does not show them scattered outside it.

        Names come from the source stem, de-duplicated with a numeric suffix because
        candidates from different sweep directories can share a basename. The mapping is also
        kept on the result, so a csv written afterwards can point at these files.
        """
        ext = SAVE_FORMATS[fmt]
        outdir = self.workdir if outdir is None else outdir
        outdir.mkdir(parents=True, exist_ok=True)
        written: dict[Path, Path] = {}
        used: set[str] = set()
        for result in self.results:
            stem = result.source.stem
            name = stem + ext
            n = 1
            while name in used:
                name = f'{stem}-{n}{ext}'
                n += 1
            used.add(name)

            atoms = result.atoms
            atoms.wrap()
            target = outdir / name
            # POSCAR lists a count per species block, so the interleaved Cl/Br from the
            # halide split would otherwise produce a header of ~30 one-atom blocks; sort to
            # group them.
            kwargs = {'sort': True, 'direct': True} if fmt == 'vasp' else {}
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                write(str(target), atoms, format=fmt, **kwargs)
            written[result.source] = target
        self.relaxed.update(written)
        return written

    def save_csv(self, path: Path | None = None) -> Path:
        """Write one row per candidate, including which ones are hull vertices.

        Goes next to the relaxed structures as hull.csv unless `path`
        says otherwise. Call it after `save_relaxed` for the `relaxed`
        column to point at the written geometries -- which is also what
        makes the directory restorable, the csv being how a later
        `HullResult(workdir=...)` finds them.
        """
        path = self.workdir / HULL_CSV if path is None else path
        lines = [
            'x,n_formula_units,energy_per_fu,energy_form,on_hull,converged,'
            'n_reducing_e_per_Y,q_Y,source,relaxed'
        ]
        for result, ef, vertex in zip(self.results, self.e_form, self.on_hull):
            n_red = result.n_reducing_e
            # Nine decimals rather than the six a formation energy is ever read to: the
            # energy column is also what a restored result reads its numbers back out of,
            # and rounding it would nudge every formation energy it then computes.
            lines.append(
                f'{result.x:.4f},{result.n_fu},{result.energy_per_fu:.9f},{ef:.9f},'
                f'{int(vertex)},{int(result.converged)},'
                f'{n_red:.4f},{3.0 - n_red:.4f},{result.source},'
                f'{self.relaxed.get(result.source, "")}'
            )
        path.write_text('\n'.join(lines) + '\n')
        return path

    def save_metadata(self, path: Path | None = None) -> Path:
        """Write everything about this result that is in neither the csv nor
        the structures.

        Namely which model produced the energies, which reference line
        they are measured against, and which optimizer settings got
        them there: the rest of what `HullResult(workdir=...)` needs to
        come back as the result that wrote it. Goes next to the relaxed
        structures as hull.json unless `path` says otherwise -- name one
        when storing a `restrict`ed window as well as the full hull, or
        the window's narrower state replaces it.
        """
        path = self.workdir / HULL_META if path is None else path
        state = {
            'model': self.model,
            'xlim': list(self.xlim) if self.xlim is not None else None,
            'reference_ends': (
                [list(end) for end in self.reference_ends]
                if self.reference_ends is not None
                else None
            ),
            'steps': self.steps,
            'fmax': self.fmax,
        }
        path.write_text(json.dumps(state, indent=2) + '\n')
        return path


class HullBuilder:
    """MLIP relaxations of ordering candidates, and the hull through them.

    This is the expensive half: it holds the settings, collects the candidate structures,
    loads the potential and relaxes each one, handing the kept relaxations to a `HullResult`
    to report on. The constructor takes exactly the options the command line exposes (see
    `build_parser`), so the CLI is a thin wrapper: `HullBuilder(**vars(args)).run()`. Each
    step is a method that reads its configuration off the instance, so they can also be
    called on their own for a single structure.
    """

    def __init__(
        self,
        *,
        inputs: list[str] | None = None,
        workdir: Path | None = None,
        model: str = 'mace',
        mace_model: str = 'medium',
        model_path: str | None = None,
        reduction_aware: bool = False,
        spin_convention: str = 'multiplicity',
        device: str = 'cpu',
        fmax: float = 0.05,
        steps: int = 500,
        relax_cell: bool = False,
        halide_samples: int = 1,
        br_fraction: float = 0.5,
        reassign_halides: bool = False,
        seed: int = 0,
        save_format: str = 'cif',
        output: Path | None = None,
        quiet: bool = False,
        progress: bool = False,
        calculator=None,
    ) -> None:
        if (model_path or reduction_aware) and model != 'mace':
            raise SystemExit('--model-path/--reduction-aware only apply to --model mace')

        self.inputs = list(inputs) if inputs is not None else []
        self.workdir = workdir
        self.model = model
        self.mace_model = mace_model
        self.model_path = model_path
        self.reduction_aware = reduction_aware
        self.spin_convention = spin_convention
        # Also normalized here, not just by argparse, so 'gpu' works in-process too.
        self.device = normalize_device(device)
        self.fmax = fmax
        self.steps = steps
        self.relax_cell = relax_cell
        self.halide_samples = halide_samples
        self.br_fraction = br_fraction
        self.reassign_halides = reassign_halides
        self.seed = seed
        self.save_format = save_format
        self.output = output
        self.quiet = quiet
        self.progress = progress
        #: A ready-made ASE calculator, used instead of loading `model`. No CLI counterpart:
        #: it is how any other potential (or a stub, in a test) gets driven from Python.
        self.calculator = calculator

        # The reduction-aware notes below are about the whole run, not one structure, so
        # they are said once however many candidates go past.
        self._redox_warned = False
        self._spin_warned = False

    def log(self, message: str = '') -> None:
        """Report progress, unless the builder was asked to keep quiet."""
        if self.quiet:
            return
        if self.progress:
            # Takes the bar down, writes above it, and puts it back, so that the builder's
            # own output scrolls past the bar instead of through it.
            tqdm.write(message)
        else:
            print(message)

    def discover_inputs(self) -> list[Path]:
        """Collect candidate structure files from `inputs` globs and/or
        `workdir`.

        A workdir is scanned for GOAC `best_*-summary.txt` files, whose
        first column lists the CIFs that were written (best-first).
        Falling back to every CIF in the directory when no summaries are
        present, so this also works on hand-made folders.
        """
        paths: list[Path] = []

        for pattern in self.inputs:
            paths.extend(Path(p) for p in sorted(glob.glob(pattern)))

        if self.workdir is not None:
            summaries = sorted(self.workdir.glob('best_*-summary.txt'))
            if summaries:
                for summary in summaries:
                    for line in summary.read_text().splitlines()[1:]:
                        fields = line.split()
                        if not fields or fields[0] == 'NaN':
                            continue
                        cif = Path(fields[0])
                        # GOAC bakes its --workdir prefix into the summary (paths relative to
                        # the sweep's CWD, not to the summary), so an absolute or CWD-relative
                        # path may not resolve here. The CIFs always sit next to the summary,
                        # so fall back to matching by basename in that directory.
                        if not cif.exists():
                            cif = summary.parent / cif.name
                        if cif.exists():
                            paths.append(cif)
            else:
                paths.extend(sorted(self.workdir.glob('*.cif')))

        # De-duplicate while preserving order.
        seen: set[Path] = set()
        unique = []
        for p in paths:
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique.append(p)
        return unique

    def load_calculator(self):
        """Return an ASE calculator for the chosen universal MLIP.

        For MACE, `model_path` loads a user-supplied checkpoint (e.g. a
        charge/spin-conditioned model fine-tuned to consume the
        `reduction_aware` state) instead of the `mace_model` foundation
        tag. When `reduction_aware` is set without such a checkpoint,
        warn that the stock foundation model is positions-only and will
        ignore the redox conditioning for its energy.
        """
        if self.calculator is not None:
            return self.calculator
        if self.model == 'mace':
            try:
                from mace.calculators import MACECalculator, mace_mp
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise SystemExit(
                    'MACE is required for --model mace. Install it with:\n'
                    '  pip install mace-torch'
                ) from exc
            if self.model_path is not None:
                # A user-trained checkpoint; if it is charge/spin-conditioned it will read
                # the reduction-aware moments/spin off each Atoms object during evaluation.
                return MACECalculator(
                    model_paths=self.model_path, device=self.device, default_dtype='float64'
                )
            if self.reduction_aware:
                warnings.warn(
                    'reduction-aware conditioning is recorded and seeds each relaxation, '
                    f'but the stock MACE-MP foundation model ({self.mace_model!r}) is '
                    'positions-only and will not use it for the energy. Only a checkpoint '
                    'fine-tuned on DFT data for this system (passed with --model-path) makes '
                    'the redox state change the energy -- the pretrained charge/spin-'
                    'conditioned models are molecular ones and do not transfer here.',
                    stacklevel=2,
                )
            return mace_mp(
                model=self.mace_model,
                default_dtype='float64',
                device=self.device,
                dispersion=False,
            )
        if self.model == 'chgnet':
            try:
                from chgnet.model.dynamics import CHGNetCalculator
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise SystemExit(
                    'CHGNet is required for --model chgnet. Install it with:\n'
                    '  pip install chgnet'
                ) from exc
            return CHGNetCalculator(use_device=self.device)
        raise SystemExit(f'unknown model {self.model!r}')  # pragma: no cover

    def halide_variants(self, atoms: Atoms) -> list[Atoms]:
        """Copies of `atoms` with a random Cl/Br split over the halide sites.

        GOAC's candidates are all-Cl; a real energy needs the 3:3 mix.
        Which halogen sits where is a separate ordering problem the
        paper leaves to DFT, so `relax_candidate` relaxes each of the
        `halide_samples` variants and keeps the best. Structures that
        already contain Br are returned as-is unless `reassign_halides`
        turns their Br back into Cl first.
        """
        if self.reassign_halides:
            atoms = _strip_br(atoms)

        symbols = np.array(atoms.get_chemical_symbols())
        halide_idx = np.where(np.isin(symbols, HALIDES))[0]
        if 'Br' in symbols[halide_idx] or len(halide_idx) == 0:
            return [atoms]

        n_br = int(round(len(halide_idx) * self.br_fraction))
        rng = np.random.default_rng(self.seed)
        variants = []
        for _ in range(self.halide_samples):
            new = atoms.copy()
            new_symbols = symbols.copy()
            new_symbols[halide_idx] = 'Cl'
            br_sites = rng.choice(halide_idx, size=n_br, replace=False)
            new_symbols[br_sites] = 'Br'
            new.set_chemical_symbols(list(new_symbols))
            variants.append(new)
        return variants

    def relax(self, atoms: Atoms, calc) -> tuple[float, bool]:
        """Relax `atoms` in place with the attached MLIP.

        Returns `(energy_eV, converged)`. A relaxation that runs out of
        `steps` still yields an energy, but an unconverged one -- too
        high by whatever strain is left in the structure. Formation
        energies here are differences of tens of meV/f.u., and the
        endpoint referencing only cancels systematic error if every
        structure is relaxed equally tightly, so a capped-out candidate
        can invent or erase a hull vertex. The flag is propagated rather
        than silently mixing the two.

        With `relax_cell` the optimizer works on the cell filter, so
        convergence accounts for the stress as well as the forces.
        """
        atoms.calc = calc
        target = CellFilter(atoms) if self.relax_cell else atoms
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            opt = FIRE(target, logfile=None)
            converged = opt.run(fmax=self.fmax, steps=self.steps)
        return float(atoms.get_potential_energy()), bool(converged)

    def relax_candidate(self, path: Path, calc) -> Result | None:
        """Relax every halide variant of one candidate and keep the lowest
        energy.

        Returns None for a structure that carries no Y, which cannot be
        placed on the composition axis at all.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            atoms = read(str(path))
        _, n_fu, x = composition_x(atoms)
        if not n_fu:
            self.log(f'  skip {path.name}: no Y, cannot assign a composition')
            return None

        variants = self.halide_variants(atoms)
        best_e = np.inf
        best_converged = False
        best_atoms = variants[0]
        redox = ''
        for variant in variants:
            if self.reduction_aware:
                n_red, q_y, spin = apply_reduction_state(
                    variant, x, spin_convention=self.spin_convention
                )
                redox = f'  [Y({q_y:+.1f}), {n_red:.1f}e/Y, spin={spin:g}]'
                if x < 3.0 and not self._redox_warned:
                    self.log('  note: x<3 oxidises the halides, not Y; reduction clamped to 0')
                    self._redox_warned = True
                if spin > TYPICAL_SPIN_EMBED_MAX and not self._spin_warned:
                    # Every Y in the cell is reduced at once, so the spin scales with the
                    # supercell: it is the honest value for this cell, but a checkpoint has
                    # to have been trained over that range for it to mean anything.
                    self.log(
                        f'  note: spin={spin:g} for {n_fu} f.u. is far outside the range '
                        f'molecular charge/spin embeddings cover '
                        f'(~{TYPICAL_SPIN_EMBED_MAX:g}); train your checkpoint with '
                        'matching --embedding_specs limits'
                    )
                    self._spin_warned = True
            energy, converged = self.relax(variant, calc)
            # An unconverged variant is only ever too high, so keeping the minimum already
            # favours the converged ones; record whether the one we kept made it.
            if energy < best_e:
                best_e, best_converged = energy, converged
                # Detach the live MLIP and freeze what it just computed, so writing this
                # structure out later cannot trigger a fresh (and expensive) evaluation.
                best_atoms = variant
                best_atoms.calc = SinglePointCalculator(
                    variant, energy=energy, forces=variant.get_forces()
                )

        e_pfu = best_e / n_fu
        flag = '' if best_converged else f'  [UNCONVERGED after {self.steps} steps]'
        self.log(
            f'  x={x:5.2f}  E={best_e:12.3f} eV  ({e_pfu:8.3f} eV/f.u.)  '
            f'{path.name}{redox}{flag}'
        )
        return Result(
            x=x,
            energy_per_fu=e_pfu,
            n_fu=n_fu,
            source=path,
            converged=best_converged,
            atoms=best_atoms,
        )

    def run(self) -> HullResult:
        """Collect the candidates, relax them all, and report on the hull."""
        paths = self.discover_inputs()
        if not paths:
            raise SystemExit('no input structures found; pass files/globs or --workdir')
        self.log(f'{len(paths)} candidate structure(s)')

        calc = self.load_calculator()

        workdir = self.output or Path(tempfile.mkdtemp(prefix='mlip_hull_'))
        workdir.mkdir(parents=True, exist_ok=True)
        self.log(f'Writing relaxed structures, hull.csv/json and hull.png to {workdir}')

        results: list[Result] = []
        bar = tqdm(
            paths,
            desc='relaxing',
            unit=' structure',
            disable=not self.progress,
            # Leaves stdout, where the builder's own output goes, to `log`.
            file=sys.stderr,
        )
        for path in bar:
            # Set before relaxing, so the bar names the structure being worked on.
            bar.set_postfix_str(path.name)
            result = self.relax_candidate(path, calc)
            if result is not None:
                results.append(result)

        if not results:
            raise SystemExit('no usable structures after relaxation')

        hull = HullResult(
            results, model=self.model, workdir=workdir, steps=self.steps, fmax=self.fmax
        )
        self.log()
        self.log(hull.summary())
        caveats = hull.caveats()
        if caveats:
            self.log(f'\n{caveats}')

        # Before the csv, so its `relaxed` column can point at the files just written.
        saved = hull.save_relaxed(fmt=self.save_format)
        n_vertices = int(hull.on_hull.sum())
        self.log(
            f'\nWrote {len(saved)} relaxed structure(s) to {workdir}/ '
            f'({n_vertices} on the hull -- marked in the table above)'
        )
        # `log` only decides whether the line is printed; the writing happens either way,
        # since the argument is evaluated before the call.
        self.log(f'Wrote {hull.save_csv()}')
        # With the csv and the structures it points at, this completes what
        # `HullResult(workdir=...)` needs to pick the run up again later.
        self.log(f'Wrote {hull.save_metadata()}')
        self.log(f'Wrote {hull.save_figure()}')
        return hull


def _strip_br(atoms: Atoms) -> Atoms:
    """Turn every Br back into Cl so `halide_variants` will re-split from
    scratch."""
    new = atoms.copy()
    new.set_chemical_symbols(['Cl' if s == 'Br' else s for s in new.get_chemical_symbols()])
    return new


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('inputs', nargs='*', help='structure files or globs (CIF/POSCAR)')
    parser.add_argument('--workdir', type=Path, help='sweep output dir to pull candidates from')
    parser.add_argument(
        '--model', choices=('mace', 'chgnet'), default='mace', help='MLIP to use'
    )
    parser.add_argument(
        '--mace-model', default='medium', help='MACE-MP foundation tag (medium/small/large)'
    )
    parser.add_argument(
        '--model-path',
        help='MACE checkpoint (e.g. charge/spin-conditioned) instead of a foundation tag',
    )
    parser.add_argument(
        '--reduction-aware',
        action='store_true',
        help='feed the Y3+ -> Y0 redox state (magnetic moments + total spin) to the potential',
    )
    parser.add_argument(
        '--spin-convention',
        choices=SPIN_CONVENTIONS,
        default='multiplicity',
        help="what goes in atoms.info['spin']: 2S+1 (MACE) or the bare moment 2S",
    )
    parser.add_argument(
        '--device',
        default='cpu',
        type=normalize_device,
        help="'cpu', 'cuda' (or 'gpu'), 'mps'",
    )
    parser.add_argument('--fmax', type=float, default=0.05, help='force convergence, eV/A')
    parser.add_argument(
        '--steps',
        type=int,
        default=500,
        help='max optimizer steps; structures hitting the cap are flagged as unconverged',
    )
    parser.add_argument('--relax-cell', action='store_true', help='relax cell as well as ions')
    parser.add_argument(
        '--halide-samples',
        type=int,
        default=1,
        help='random Cl/Br arrangements to try per all-Cl candidate (best kept)',
    )
    parser.add_argument(
        '--br-fraction', type=float, default=0.5, help='fraction of halide sites made Br'
    )
    parser.add_argument(
        '--reassign-halides',
        action='store_true',
        help='reshuffle Cl/Br even when the input already contains Br',
    )
    parser.add_argument('--seed', type=int, default=0, help='seed for halide arrangements')
    parser.add_argument(
        '--output',
        type=Path,
        metavar='DIR',
        help='directory for the relaxed structures (for visualization in VESTA/OVITO/ASE), '
        'hull.csv, hull.json and hull.png; a fresh temp directory by default, and what '
        'HullResult(workdir=...) reads a finished run back out of',
    )
    parser.add_argument(
        '--save-format',
        choices=tuple(SAVE_FORMATS),
        default='cif',
        help='format for the relaxed structures; extxyz also carries energy and magnetic '
        'moments',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help="don't report progress; the structures, hull.csv and hull.png are still written",
    )
    parser.add_argument(
        '--progress',
        action='store_true',
        help='draw a progress bar over the candidate structures on stderr',
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    # The parser's dest names match HullBuilder's constructor one-for-one.
    HullBuilder(**vars(args)).run()


if __name__ == '__main__':
    main()
