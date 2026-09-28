"""Screen orderings of a mobile species with GOAC's Coulomb (Ewald) sum.

Works on any partially-occupied CIF. The positions that hold the mobile species
(`--specie`, Li by default) become one group that GOAC permutes the species over, and the
script sweeps the composition x, the number of mobile ions per formula unit of the rest of
the structure (so x in Li_x YCl3Br3, or Na_x SbS4). At each x it samples orderings and plots
the lowest Coulomb energy per formula unit over the cloud of sampled orderings, with a
dotted tie line between the end members so the deviation in between can be read off. The
lowest-energy orderings are written as CIFs, ready to be relaxed with DFT or an MLIP
(`mlip_hull.py`).

This is the Coulomb pre-screening step of

    Beneficial redox activity of halide solid electrolytes empowering high-performance
    anodes in all-solid-state batteries, https://doi.org/10.26434/chemrxiv-2024-x2rld

where the Li sublattice of Li_x YCl3Br3 was screened with "100,000 permutations of Li
distribution in interstitial positions at each Li concentration"; that material, gemdat's
bundled `Li3YCl3Br3-c2m.cif`, is the worked example below.

How the sites of the CIF are treated:

* *Mobile* positions: every site holding `--specie`. Other elements sharing those
  positions (Y on Li2/Y9 in Li3YCl3Br3) are the *partners*.
* *Partner* positions: sites holding only partner elements (Y1). Without
  `--disorder-partners` they are filled with their majority element and only the mobile
  species is permuted; with it the partners join the mobile group and are permuted too.
* *Framework*: everything else, kept as it is. A framework site shared by elements of
  equal charge (Cl/Br) is collapsed onto the lightest one, since to a point-charge model
  they are indistinguishable and the ordering cannot change the energy -- settle it with
  DFT instead. A framework site with mixed charges or vacancies stays partially occupied,
  so GOAC permutes it as a group of its own.

Charges default to pymatgen's oxidation-state guess for the CIF's composition; set them
with `--charges Li=1 Y=3 Cl=-1 Br=-1`. The formula unit defaults to the reduced formula
of everything but the mobile species; override with `--formula-units`.

Note that this is a *Coulomb* energy with fixed formal charges. Away from charge
neutrality the cell carries a net charge and the redox that compensates it has no
counterpart in the model, so the curve is a candidate-generation step, not a stability
prediction. What GOAC does settle is the ordering problem at each *fixed* composition.

How far x can go is set by the mobile positions the CIF lists; `--x-max` is clamped to
that. `--extra-interstitials` searches the anion packing for the remaining voids and adds
them as mobile positions (complete Wyckoff orbits, thresholds from
`--void-radius`/`--void-separation`).

GOAC is an optional dependency:

    pip install GOAC --find-links https://github.com/GEMDAT-repos/GOAC/releases/expanded_assets/0.1.1

Examples
--------
    python scripts/goac_sweep.py my_structure.cif --specie Na --supercell 2 2 2
    python scripts/goac_sweep.py src/gemdat/data/Li3YCl3Br3-c2m.cif \
        --supercell 2 1 2 --extra-interstitials --x-max 9 --n-best 10 --output out/

Programmatic use mirrors the CLI, split in two: a `GOACSweep` holds the settings and does
the sampling, and hands back a `GOACResult` holding what it sampled, which does the
reporting on it. The sweep's constructor takes the same options as the argument parser
(`GOACSweep(**vars(args))` is exactly what `main` does), so

    result = GOACSweep(cif=DATA / 'Li3YCl3Br3-c2m.cif', x_max=4).run()
    result.scatter_max = 200
    result.save_figure(Path('sparser.png'))

`run` reports as it goes unless `quiet=True` (`--quiet`), and `progress=True`
(`--progress`) puts a bar over the compositions on stderr. The individual setup steps
(`build_input_structure`, `sample_energies`, `add_void_interstitials`, ...) are methods
on the sweep, reading their configuration off it.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import tempfile
import warnings
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from pymatgen.core import Composition, Element, Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm

# GOAC is imported where it is used, so this module (and
# `calculate_coarse_potentials.py`) imports without it.
GOAC_MISSING = (
    'GOAC is required for this. Install it with:\n'
    '  pip install GOAC --find-links '
    'https://github.com/GEMDAT-repos/GOAC/releases/expanded_assets/0.1.1'
)

# Energy window, in eV, within which two orderings count as the same solution when
# collecting the n best. Symmetry-equivalent arrangements have *identical* Coulomb
# energies, so anything above numerical noise separates them; 1 meV is well below the
# resolution at which the subsequent DFT relaxation would tell two candidates apart.
# GOAC's own default is 1e-8, which keeps the duplicates; -1 disables filtering entirely.
DEFAULT_TOL = 1e-3

# GOAC solver modes. 'random' is what the paper did (draw N permutations, keep the best);
# the others seed from those draws and then optimize, which matters for large supercells.
SOLVERS = {
    'random': 'Random',
    'sa': 'Random-SA',
    'mc': 'Random-MC',
    'ga': 'Random-GA',
}

Coords = list[np.ndarray]


class Table(str):
    def __repr__(self) -> str:
        return str(self)

    def _repr_pretty_(self, printer, cycle) -> None:
        printer.text(str(self))


@dataclass
class SweepPoint:
    """What the sweep found at one composition."""

    x: float
    #: Coulomb energies of the randomly sampled orderings — the cloud behind the curve.
    energies: np.ndarray
    #: The lowest energy found, by the solver or by the sampling, whichever went deeper.
    best_energy: float
    #: CIFs written for this composition, best-first, as (path, energy) pairs.
    structures: list[tuple[Path, float]]


@dataclass
class GOACResult:
    """What a `GOACSweep` sampled, and the reporting on it.

    The sweep hands one of these back from `run`. It owns everything
    that reads the sampled energies rather than produces them, so a
    finished sweep can be tabulated and plotted again, with different
    settings, without paying for the sampling twice.
    """

    points: list[SweepPoint]
    #: Formula units in the supercell GOAC summed over; energies are reported per f.u.
    n_formula_units: int
    #: Where the sweep wrote its CIFs, and where `save_figure` puts the figure by default.
    workdir: Path
    #: Most sampled configurations to scatter per composition: 100k points per composition
    #: make for a huge, unreadable figure.
    scatter_max: int = 2000
    #: What x counts, for the axis label.
    x_label: str = 'x'

    @property
    def xs(self) -> np.ndarray:
        """The compositions sampled, in sweep order."""
        return np.array([point.x for point in self.points])

    @property
    def e_min(self) -> np.ndarray:
        """Lowest energy per formula unit, as mlip_hull.py reports it too."""
        energies = np.array([point.best_energy for point in self.points])
        return energies / self.n_formula_units

    def summary(self) -> Table:
        """The lowest energy against composition, as a printable table."""
        lines = ['  x      E_min (eV/f.u.)']
        lines += [f'  {x:5.2f}  {energy:14.3f}' for x, energy in zip(self.xs, self.e_min)]
        return Table('\n'.join(lines))

    def plot(self) -> Figure:
        """Lowest Coulomb energy against composition, over the sampled cloud.

        As in Fig. 2a of the paper, where every configuration is plotted
        and the line traces the minimum, plus a tie line between the end
        members so the deviation in between can be read off directly.
        """
        fig, ax = plt.subplots(figsize=(6, 4))

        rng = np.random.default_rng(0)
        for point in self.points:
            energies = point.energies
            if len(energies) > self.scatter_max:
                energies = rng.choice(energies, size=self.scatter_max, replace=False)
            ax.scatter(
                np.full(len(energies), point.x),
                energies / self.n_formula_units,
                s=1,
                color='lightgrey',
                zorder=1,
                label='candidates' if point is self.points[0] else None,
            )

        xs, e_min = self.xs, self.e_min
        ax.plot(
            [xs[0], xs[-1]],
            [e_min[0], e_min[-1]],
            ':',
            color='black',
            zorder=2,
            label='end-member tie line',
        )
        ax.plot(xs, e_min, 'o-', color='crimson', zorder=3, label='lowest energy')
        ax.set_xlabel(self.x_label)
        ax.set_ylabel('Coulomb energy (eV/f.u.)')
        ax.set_title('Lowest Coulomb energy versus composition')
        ax.legend()

        fig.tight_layout()
        return fig

    def save_figure(self, path: Path | None = None) -> Path:
        """Write `plot` next to the CIFs as sweep.png, or wherever `path`
        says."""
        figure = path if path is not None else self.workdir / 'sweep.png'
        self.plot().savefig(figure, dpi=150)
        return figure


class GOACSweep:
    """A GOAC Coulomb pre-screening sweep over x in `specie`_x (rest of the
    CIF).

    This is the setup half: it holds the settings, builds the partially-occupied cells and
    runs GOAC over them, handing the sampled energies to a `GOACResult` to report on. The
    constructor takes exactly the options the command line exposes (see `build_parser`), so
    the CLI is a thin wrapper: `GOACSweep(**vars(args)).run()`. Each step of the sweep is a
    method that reads its configuration off the instance, so they can also be called on
    their own for a single composition.
    """

    def __init__(
        self,
        *,
        cif: Path,
        specie: str = 'Li',
        charges: dict[str, float] | list[tuple[str, float]] | None = None,
        formula_units: int | None = None,
        supercell: list[int] | None = None,
        samples: int = 100000,
        solver: str = 'sa',
        steps: int = 200000,
        starts: int = 20,
        disorder_partners: bool = False,
        extra_interstitials: bool = False,
        void_radius: float = 2.3,
        void_separation: float = 2.2,
        n_best: int = 1,
        tol: float = DEFAULT_TOL,
        output: Path | None = None,
        x_min: float = 0.0,
        x_max: float | None = None,
        x_step: float = 0.5,
        scatter_max: int = 2000,
        quiet: bool = False,
        progress: bool = False,
    ) -> None:
        self.cif = Path(cif)
        self.specie = specie
        # Both are guessed from the CIF by `resolve` when not given.
        # A dict, or the (element, charge) pairs the command line parses.
        self.charges = dict(charges) if charges else None
        self.formula_units = formula_units
        # A fresh list avoids the mutable-default trap.
        self.supercell = list(supercell) if supercell is not None else [1, 1, 1]
        self.samples = samples
        self.solver = solver
        self.steps = steps
        self.starts = starts
        self.disorder_partners = disorder_partners
        self.extra_interstitials = extra_interstitials
        self.void_radius = void_radius
        self.void_separation = void_separation
        self.n_best = n_best
        self.tol = tol
        self.output = output
        self.x_min = x_min
        self.x_max = x_max
        self.x_step = x_step
        self.scatter_max = scatter_max
        self.quiet = quiet
        self.progress = progress

    def resolve(self, base: Structure) -> None:
        """Fill in `charges` and `formula_units` from `base`, where not given.

        Charges are pymatgen's most likely oxidation states for the
        composition, which needs it to be integer and charge-balanced.
        The formula unit is the reduced formula of everything but the
        mobile species, so that x counts mobile ions per formula unit of
        the host. When that composition is fractional (Sb1.7 W0.3 S8),
        the positions are counted instead, per distinct site
        composition.
        """
        if self.charges is None:
            try:
                guesses = base.composition.oxi_state_guesses(max_sites=-1)
            except ValueError:
                guesses = ()
            if not guesses:
                raise ValueError(
                    f'cannot guess oxidation states for {base.composition.formula}; '
                    'pass the charges explicitly (--charges EL=Q ...)'
                )
            self.charges = dict(guesses[0])
        if self.formula_units is None:
            amounts = [a for el, a in base.composition.items() if el.symbol != self.specie]
            if all(abs(a - round(a)) < 1e-3 for a in amounts):
                counts = [round(a) for a in amounts]
            else:
                partner_only, _, framework = self.split_sites(base)
                sites = [str(site.species) for site in partner_only + framework]
                counts = [sites.count(key) for key in set(sites)]
            self.formula_units = math.gcd(*counts) or 1

    def host_formula(self, base: Structure) -> str:
        """Formula of one formula unit of everything but the mobile species."""
        assert self.formula_units is not None
        host = {
            element: amount / self.formula_units
            for element, amount in base.composition.items()
            if element.symbol != self.specie
        }
        return Composition(host).formula.replace(' ', '')

    def goac_charges(self) -> dict[str, tuple[float, float]]:
        """`charges` as GOAC wants them.

        The keys must be <element><wildcard>: GOAC rebuilds each label
        from the site's element symbol plus the trailing number of the
        CIF label, so a key that does not start with a real element
        symbol silently matches nothing and leaves those ions with
        charge 0.
        """
        assert self.charges is not None
        return {f'{element}*': (float(q), 0.1) for element, q in self.charges.items()}

    def log(self, message: str = '') -> None:
        """Report progress, unless the sweep was asked to keep quiet."""
        if self.quiet:
            return
        if self.progress:
            # Takes the bar down, writes above it, and puts it back, so that the sweep's
            # own output scrolls past the bar instead of through it.
            tqdm.write(message)
        else:
            print(message)

    @contextmanager
    def silence(self) -> Iterator[None]:
        """Swallow GOAC's own output for the duration of the block.

        A no-op unless the sweep is quiet or drawing a progress bar,
        which needs the terminal to itself. GOAC narrates every problem
        and every solver run, partly from Python and partly from the
        Fortran core, and the two need catching differently:
        `sys.stdout` is rebound for the Python prints, and file
        descriptor 1 -- which the Fortran core writes to directly,
        behind Python's back -- is pointed at the same temporary file.
        Neither alone is enough. Under Jupyter they are not even the
        same stream: the kernel sends `sys.stdout` to the notebook over
        its own socket, so a descriptor swap on its own leaves GOAC's
        Python prints in the cell output.

        What was captured is replayed on stderr if the block raises, so
        a failing run still says why it failed.

        Fortran buffers its writes, so a tail of solver output can
        surface after the descriptor is restored; keeping one block
        around the whole GOAC call rather than one per call keeps that
        to the end of the sweep at worst.
        """
        if not (self.quiet or self.progress):
            yield
            return

        failed = False
        with tempfile.TemporaryFile('w+') as sink:
            sys.stdout.flush()
            saved = os.dup(1)
            try:
                os.dup2(sink.fileno(), 1)
                with redirect_stdout(sink):
                    yield
            except BaseException:
                failed = True
                raise
            finally:
                sys.stdout.flush()
                os.dup2(saved, 1)
                os.close(saved)
                if failed:
                    sink.seek(0)
                    sys.stderr.write(sink.read())

    def split_sites(self, structure: Structure) -> tuple[list, list, list]:
        """Partner-only, mobile and framework sites of `structure`.

        Mobile sites hold `specie`; partners are the other elements
        sharing those sites (in Li3YCl3Br3 the refinement puts Y on both
        2a and on 4h, and pymatgen merges the 4h one with Li2 into one
        partially occupied Li/Y site), and partner-only sites hold
        nothing but partners. Everything else is framework.
        """
        mobile = [site for site in structure if self.specie in site.species.get_el_amt_dict()]
        partners = {el for site in mobile for el in site.species.get_el_amt_dict()}
        partners.discard(self.specie)

        partner_only: list = []
        framework: list = []
        for site in structure:
            elements = set(site.species.get_el_amt_dict())
            if self.specie in elements:
                continue
            if partners and elements <= partners:
                partner_only.append(site)
            else:
                framework.append(site)
        return partner_only, mobile, framework

    def partner_counts(self, structure: Structure) -> dict[str, int]:
        """Number of each partner ion in the cell, rounded from the refined
        occupancies."""
        partner_only, mobile, _ = self.split_sites(structure)
        totals: dict[str, float] = {}
        for site in partner_only + mobile:
            for element, amount in site.species.get_el_amt_dict().items():
                if element != self.specie:
                    totals[element] = totals.get(element, 0.0) + amount
        return {element: int(round(amount)) for element, amount in totals.items()}

    def mobile_positions_per_cell(self, structure: Structure) -> int:
        """How many positions per cell the mobile species can be spread over.

        With `disorder_partners` that is every mobile and partner
        position minus the ones the partner ions occupy, which share the
        group; without it the partners sit apart on their own positions
        and only the mobile positions are free.
        """
        partner_only, mobile, _ = self.split_sites(structure)
        if self.disorder_partners:
            taken = sum(self.partner_counts(structure).values())
            return len(partner_only) + len(mobile) - taken
        return len(mobile)

    def mean_charge(self, site) -> float:
        """Occupancy-weighted formal charge of `site`."""
        assert self.charges is not None
        return sum(
            self.charges[element] * amount
            for element, amount in site.species.get_el_amt_dict().items()
        )

    @staticmethod
    def _orbit(point: np.ndarray, ops: list, lattice: Lattice, tol: float) -> np.ndarray:
        """Symmetry orbit of `point`, with images closer than `tol` (in A)
        merged.

        A grid point never sits exactly on a special position, so its
        images scatter by up to the grid spacing instead of coinciding;
        merged images are averaged, which snaps the site back onto the
        symmetry element it belongs to.
        """
        images = np.array([op.operate(point) % 1.0 for op in ops])
        close = lattice.get_all_distances(images, images) < tol

        # Every image joins the group of the earliest image it touches. Leaders are
        # resolved in order, so one hop is enough to land on the group's representative.
        leader = np.arange(len(images))
        for i in range(len(images)):
            leader[i] = leader[np.argmax(close[i, : i + 1])]

        orbit = []
        for group in (images[leader == i] for i in np.unique(leader)):
            # Bring the group into the representative's periodic image before averaging, so
            # that e.g. 0.999 and 0.001 do not average to 0.5.
            orbit.append((group[0] + (group - group[0] + 0.5) % 1.0 - 0.5).mean(axis=0) % 1.0)
        return np.array(orbit)

    def find_void_interstitials(
        self, base: Structure, *, grid_spacing: float = 0.2
    ) -> np.ndarray:
        """Empty interstitial voids of the anion sublattice, as fractional
        coords.

        The CIF only lists the mobile positions the refinement could resolve, which caps
        how much of the mobile species can be inserted (see `--x-max`). This finds the
        remaining holes in the anion packing so that higher x becomes reachable: every
        point of a grid over the cell that is at least `void_radius` from any anion and
        `void_separation` from any site already in the structure is a candidate, and the
        deepest ones are picked greedily, each expanded to its full symmetry orbit so that
        the added positions form complete Wyckoff sets rather than an arbitrary
        symmetry-broken subset. An orbit is dropped if it collides with itself or with what
        has already been accepted.

        Reasonable thresholds come from the sites the refinement *did* resolve: in
        Li3YCl3Br3 those sit 2.30-2.75 A from the nearest halide, and no two cation
        positions are closer than 2.22 A.
        """
        self.resolve(base)
        min_anion_dist = self.void_radius
        min_separation = self.void_separation

        anions = np.array([site.frac_coords for site in base if self.mean_charge(site) < 0])
        cations = np.array([site.frac_coords for site in base if self.mean_charge(site) >= 0])
        if not len(anions):
            raise ValueError('no anions to find voids between: check the charges')
        occupied = np.array([site.frac_coords for site in base])
        lattice = base.lattice

        divisions = np.maximum(np.ceil(np.array(lattice.abc) / grid_spacing).astype(int), 1)
        grid = np.stack(
            np.meshgrid(*[np.arange(n) / n for n in divisions], indexing='ij'), axis=-1
        ).reshape(-1, 3)

        # Anions first: that thins the grid by orders of magnitude, so the second pass
        # runs over a handful of points instead of the whole cell.
        clearance = lattice.get_all_distances(grid, anions).min(axis=1)
        keep = clearance >= max(min_anion_dist, min_separation)
        grid, clearance = grid[keep], clearance[keep]
        if len(cations):
            keep = lattice.get_all_distances(grid, cations).min(axis=1) >= min_separation
            grid, clearance = grid[keep], clearance[keep]

        # Deepest voids first, so each orbit is seeded by the most open point in it.
        grid = grid[np.argsort(-clearance)]

        ops = SpacegroupAnalyzer(base).get_symmetry_operations()
        merge_tol = min(1.0, min_separation / 2)

        orbits: list[np.ndarray] = []
        taken = occupied
        cursor = 0
        while cursor < len(grid):
            orbit = self._orbit(grid[cursor], ops, lattice, merge_tol)
            cursor += 1
            if lattice.get_all_distances(orbit, taken).min() < min_separation:
                continue
            if len(orbit) > 1:
                within = lattice.get_all_distances(orbit, orbit)
                np.fill_diagonal(within, np.inf)
                if within.min() < min_separation:
                    continue
            orbits.append(orbit)
            taken = np.vstack([taken, orbit])
            # Drop the points the new orbit swallows in one pass, rather than rejecting
            # them one at a time on the way past.
            grid = grid[cursor:]
            if len(grid):
                far = lattice.get_all_distances(grid, orbit).min(axis=1) >= min_separation
                grid = grid[far]
            cursor = 0

        return np.vstack(orbits) if orbits else np.zeros((0, 3))

    def add_void_interstitials(self, base: Structure) -> tuple[Structure, int]:
        """`base` with the voids found by `find_void_interstitials` added as
        mobile sites.

        Only the *positions* of the added sites matter:
        `build_input_structure` assigns every mobile position a fresh
        occupancy from the requested x, and reads the partner counts off
        the sites the refinement provided, none of which are touched
        here.
        """
        voids = self.find_void_interstitials(base)
        structure = base.copy()
        for frac in voids:
            structure.insert(len(structure), self.specie, frac, label=f'{self.specie}_void')
        return structure, len(voids)

    def build_input_structure(self, base: Structure, *, x: float) -> Structure:
        """Build the partially-occupied unit cell GOAC iterates over.

        The mobile species is spread over its positions as a single uniform partial
        occupancy, so that GOAC groups them into one iterative site and permutes the ions
        freely across all of them -- the paper's "Li distribution in interstitial
        positions".

        Occupancies are per unit cell on purpose: GOAC builds the supercell itself after
        reading the CIF, and derives the ion count per site group from occupancy times the
        number of positions in that supercell. Passing absolute counts here would make the
        composition depend on the supercell.

        With `disorder_partners` the partner ions join the same group and are permuted
        along with the mobile species. Without it, each partner-only position is filled
        with its majority element and only the mobile species is permuted.

        Framework sites shared by elements of one charge are collapsed onto the lightest
        of them: GOAC treats them as constants, and the point-charge energy cannot tell
        the difference. Framework sites with mixed charges or vacancies stay partially
        occupied, one GOAC group per distinct composition.
        """
        self.resolve(base)
        assert self.charges is not None and self.formula_units is not None
        partner_only, mobile, framework = self.split_sites(base)

        species: list[dict[str, float]] = []
        coords: Coords = []
        labels: list[str] = []

        def add(amounts: dict[str, float], frac: np.ndarray, label: str) -> None:
            species.append(amounts)
            coords.append(frac)
            labels.append(label)

        group = {self.specie: x * self.formula_units}
        positions = mobile
        if self.disorder_partners:
            positions = partner_only + mobile
            group.update(self.partner_counts(base))
        if sum(group.values()) > len(positions) + 1e-9:
            wanted = ', '.join(f'{n:g} {element}' for element, n in group.items())
            raise ValueError(
                f'x={x:g} needs {wanted}, but there are only {len(positions)} positions '
                'per cell'
            )
        occupancies = {element: n / len(positions) for element, n in group.items()}
        for site in positions:
            add(occupancies, site.frac_coords, f'{self.specie}1')

        # Labels only need to be unique, apart from the ones shared on purpose by a group.
        number = 2
        if not self.disorder_partners:
            for site in partner_only:
                amounts = site.species.get_el_amt_dict()
                element = max(amounts, key=amounts.get)
                add({element: 1.0}, site.frac_coords, f'{element}{number}')
                number += 1

        groups: dict[tuple, str] = {}
        for site in framework:
            amounts = site.species.get_el_amt_dict()
            one_charge = len({self.charges[element] for element in amounts}) == 1
            if one_charge and sum(amounts.values()) > 0.999:
                element = min(amounts, key=lambda el: Element(el).Z)
                add({element: 1.0}, site.frac_coords, f'{element}{number}')
                number += 1
                continue
            key = tuple(sorted(amounts.items()))
            if key not in groups:
                groups[key] = f'{key[0][0]}{number}'
                number += 1
            add(amounts, site.frac_coords, groups[key])

        return Structure(base.lattice, species, coords, labels=labels)

    def sample_energies(
        self, structure: Structure, *, workdir: Path, tag: str
    ) -> tuple[np.ndarray, float, list[tuple[Path, float]]]:
        """Optimize the cation ordering with GOAC.

        Returns the energies of `samples` random configurations (the landscape), the lowest
        energy found by the chosen solver, and up to `n_best` lowest-energy structures it
        wrote, as (path, energy) pairs ordered best-first. Structures within `tol` eV of one
        already kept are dropped (see DEFAULT_TOL), so fewer than `n_best` may come back.

        The two differ once the cell gets big: in a 2x1x2 supercell of Li3YCl3Br3 there are
        ~1e18 ways to place the Li ions, so 100,000 random draws land well above the true
        minimum. The paper stopped at random permutations and handed the best ones to DFT;
        the annealing and genetic solvers below dig considerably deeper for the same cost.
        """
        import GOAC
        from GOAC.IterationProblem import Iteration_Problem
        from GOAC.RandomSolver import Random_Solver

        solver_name = SOLVERS[self.solver]

        # GOAC reports on every problem and every solver run of its own accord; `silence`
        # is what makes --quiet quiet, and does nothing otherwise.
        with self.silence():
            cif_path = workdir / f'input_{tag}.cif'
            with warnings.catch_warnings():
                # Non-unique site labels are deliberate: that is how GOAC groups the
                # positions a species is permuted over into one iterative site.
                warnings.simplefilter('ignore')
                structure.to(filename=str(cif_path))
                problem = Iteration_Problem(
                    cif_file=str(cif_path),
                    fixed_sites=[],
                    charges=self.goac_charges(),
                    supercell=self.supercell,
                )
            problem.calc_coulomb_matrices()

            if not problem.iterate_sites:
                # Fully ordered (x=0 or every interstitial filled): a single configuration.
                energy = float(problem.const)
                return np.array([energy]), energy, []

            # For 'Random', this is simply how many permutations to draw. For the optimizing
            # solvers it is the number of random *starting points*, each of which is then run
            # for the full step budget — passing `samples` there would cost samples x steps
            # moves.
            n_runs = self.samples if solver_name == 'Random' else self.starts
            if self.n_best > n_runs and solver_name != 'Random':
                # The optimizing solvers only ever have `n_runs` finished runs to rank, so
                # they cannot produce more distinct structures than they had starting points.
                warnings.warn(
                    f'--n-best {self.n_best} exceeds --starts {n_runs}; at most {n_runs} '
                    f'structures can be written by {solver_name}',
                    stacklevel=2,
                )

            # `w` is how many of the ranked solutions GOAC writes out as CIFs.
            solver = Random_Solver(name=solver_name, problem=problem, n=n_runs, w=self.n_best)
            solver.initialize()
            solver.opt['tol'] = self.tol
            if self.steps is not None:
                if solver_name in ('Random-SA', 'Random-MC'):
                    solver.opt['mc_steps'] = self.steps
                    # GOAC's defaults assume 1e7 steps; rescale the cooling schedule so the
                    # anneal still finishes cold within whatever budget was asked for.
                    solver.opt['mc_sim_an_steps'] = max(1, self.steps // 200)
                    solver.opt['mc_write_steps'] = self.steps + 1
                elif solver_name == 'Random-GA':
                    solver.opt['ga_steps'] = self.steps

            out_name = str(workdir / f'best_{tag}')
            solver.solve(out_name)
            best = self.read_summary(Path(f'{out_name}-summary.txt'))

            # The landscape itself always comes from plain random sampling, so that the
            # histogram means the same thing whichever solver was used to find the minimum.
            energies, _ = GOAC.random_samples(
                solver.species_nums,
                solver.species_occs,
                solver.shared_sites,
                problem.const,
                solver.np_alpha,
                problem.beta,
                self.samples,
            )
        energies = np.asarray(energies, dtype=float)
        return energies, min(best[0][1], float(energies.min())), best

    @staticmethod
    def read_summary(path: Path) -> list[tuple[Path, float]]:
        """Structures listed in a GOAC `-summary.txt` file, which is sorted
        best-first.

        Rows whose CIF was not written (GOAC prints `NaN` past its write
        limit, and a trailing note once it runs out of distinct
        solutions) are skipped.
        """
        rows = []
        for line in path.read_text().splitlines()[1:]:
            fields = line.split()
            if len(fields) < 2 or fields[0] == 'NaN':
                continue
            try:
                energy = float(fields[1])
            except ValueError:
                continue
            cif = Path(fields[0])
            if cif.exists():
                rows.append((cif, energy))
        if not rows:
            raise ValueError(f'no structures found in {path}')
        return rows

    def run_sweep(self, base: Structure, workdir: Path) -> GOACResult:
        """Sample every composition in the range, reporting progress as it
        goes."""
        self.resolve(base)
        assert self.formula_units is not None
        cells = int(np.prod(self.supercell))
        n_fu = self.formula_units * cells
        per_cell = self.mobile_positions_per_cell(base)
        x_max = per_cell / self.formula_units
        if self.x_max is not None:
            x_max = min(self.x_max, x_max)

        xs = np.arange(self.x_min, x_max + 1e-9, self.x_step)
        if not len(xs):
            raise SystemExit(f'--x-min {self.x_min:g} is above the reachable x <= {x_max:g}')

        self.log(f'Sweeping x = {xs[0]:g} .. {xs[-1]:g} in steps of {self.x_step:g}')
        self.log(f'Supercell {self.supercell}, {self.samples} random configurations per point')
        self.log(
            f'{per_cell * cells} {self.specie} positions for {n_fu} formula units '
            f'of {self.host_formula(base)}\n'
        )

        points = []
        bar = tqdm(
            xs,
            desc='sweeping',
            unit=' composition',
            disable=not self.progress,
            # Leaves stdout, where the sweep's own output goes, to `log`.
            file=sys.stderr,
        )
        for x in bar:
            # Set before sampling, so the bar names the composition being worked on.
            bar.set_postfix_str(f'x={x:g}')
            structure = self.build_input_structure(base, x=x)
            energies, best_energy, best = self.sample_energies(
                structure,
                workdir=workdir,
                tag=f'x{x:g}',
            )
            points.append(
                SweepPoint(
                    x=float(x),
                    energies=energies,
                    best_energy=best_energy,
                    structures=best,
                )
            )

            if self.quiet:
                continue

            n_mobile = int(round(x * n_fu))
            cif = f'  {best[0][0].name}' if best else ''
            self.log(
                f'  x={x:5.2f}  n_{self.specie}={n_mobile:3d}  '
                f'E_min={best_energy:10.3f} eV{cif}'
            )
            for rank, (cif_path, energy) in enumerate(best):
                candidate = Structure.from_file(cif_path)
                self.log(
                    f'    {rank}: {energy:10.3f} eV  {candidate.composition.formula}'
                    f'  {cif_path.name}'
                )

        return GOACResult(
            points=points,
            n_formula_units=n_fu,
            workdir=workdir,
            scatter_max=self.scatter_max,
            x_label=f'x in {self.specie}$_x${self.host_formula(base)}',
        )

    def run(self) -> GOACResult:
        """Load the CIF, add interstitials if asked, and run the whole
        sweep."""
        if importlib.util.find_spec('GOAC') is None:
            raise SystemExit(GOAC_MISSING)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            base = Structure.from_file(self.cif)
        self.log(f'Loaded {self.cif}: {base.composition.formula}, {len(base)} sites')
        # Before any voids are added, which would throw the oxidation-state guess off.
        self.resolve(base)
        self.log(f'Charges {self.charges}, {self.formula_units} formula units per cell')

        assert self.formula_units is not None
        if self.extra_interstitials:
            base, n_added = self.add_void_interstitials(base)
            available = self.mobile_positions_per_cell(base)
            self.log(
                f'Added {n_added} void interstitial positions: {available} positions per '
                f'cell available to {self.specie}, so x <= {available / self.formula_units:g}'
            )

        workdir = self.output or Path(tempfile.mkdtemp(prefix='goac_sweep_'))
        workdir.mkdir(parents=True, exist_ok=True)
        self.log(f'Writing structures and sweep.png to {workdir}')

        result = self.run_sweep(base, workdir)
        # Outside the log call: the figure is written either way, quiet or not.
        figure = result.save_figure()
        self.log()
        self.log(result.summary())
        self.log(f'Wrote {figure}')
        return result


class DefaultsFormatter(argparse.HelpFormatter):
    """Help formatter that puts the default in a column of its own.

    `argparse.ArgumentDefaultsHelpFormatter` appends `(default: ...)` to
    the help text, where it wraps along with the prose; this puts it in
    front instead, aligned, so the defaults can be read straight down.
    Options with no default are left alone, since `(default: None)` says
    nothing — unless they name what happens instead, by carrying a
    `default_text` attribute (see `--output`).
    """

    # Joins the default to the help text, so that _split_lines can take the two apart
    # again after argparse has expanded and wrapped what _get_help_string returned.
    _SEP = '\x00'
    # Widest default column tolerated; anything longer keeps its own line.
    _MAX_COLUMN = 16

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._column = 0

    def add_arguments(self, actions) -> None:
        # Called for every argument group before anything is rendered, so the column is
        # sized to the widest default by the time _split_lines needs it.
        for action in actions:
            default = self._default_text(action)
            if default:
                self._column = max(self._column, min(len(default) + 2, self._MAX_COLUMN))
        super().add_arguments(actions)

    def _default_text(self, action) -> str:
        """`action`'s default as it is shown, or '' if it has none to show."""
        if not action.help or action.default is argparse.SUPPRESS:
            return ''
        stated = getattr(action, 'default_text', None)
        if stated:
            return f'[{stated}]'
        if action.default is None:
            return ''
        if isinstance(action.default, (list, tuple)):
            # As it would be typed on the command line, not as a Python literal.
            return '[{}]'.format(' '.join(str(v) for v in action.default))
        if isinstance(action.default, Path):
            # Relative to where the script is being run from, when that is shorter than
            # the absolute path the default was built from.
            try:
                return f'[{action.default.relative_to(Path.cwd())}]'
            except ValueError:
                return f'[{action.default}]'
        return f'[{action.default}]'

    def _get_help_string(self, action) -> str | None:
        default = self._default_text(action)
        if not default:
            return action.help
        # `%` is the escape character of the expansion argparse runs over this text.
        return f'{default.replace("%", "%%")}{self._SEP}{action.help}'

    def _split_lines(self, text, width) -> list[str]:
        head, sep, help_text = text.partition(self._SEP)
        if not sep:
            return super()._split_lines(text, width)
        pad = self._column
        lines = super()._split_lines(help_text, max(width - pad, 20))
        if len(head) >= pad:
            # Too wide to share a line with the help text; give it one to itself.
            return [head, *(' ' * pad + line for line in lines)]
        return [head.ljust(pad) + lines[0], *(' ' * pad + line for line in lines[1:])]


def parse_charges(text: str) -> tuple[str, float]:
    """Parse one `--charges` entry, `EL=Q`."""
    element, sep, charge = text.partition('=')
    if not sep:
        raise argparse.ArgumentTypeError(f'expected EL=Q, got {text!r}')
    return element, float(charge)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=DefaultsFormatter
    )
    parser.add_argument('cif', type=Path, help='input CIF, may be partially occupied')
    parser.add_argument('--specie', default='Li', help='mobile species to order')
    parser.add_argument(
        '--charges',
        type=parse_charges,
        nargs='+',
        metavar='EL=Q',
        help='formal charges, e.g. Li=1 Y=3 Cl=-1 Br=-1 (default: guessed from the CIF)',
    )
    parser.add_argument(
        '--formula-units',
        type=int,
        help='formula units per cell that x is counted per (default: from the reduced '
        'formula of everything but --specie)',
    )
    parser.add_argument(
        '--supercell',
        type=int,
        nargs=3,
        default=[1, 1, 1],
        metavar=('NA', 'NB', 'NC'),
        help='supercell used for the Coulomb sum (the Li3YCl3Br3 paper uses 2 1 2)',
    )
    parser.add_argument(
        '--samples', type=int, default=100000, help='random orderings per composition'
    )
    parser.add_argument(
        '--solver',
        choices=tuple(SOLVERS),
        default='sa',
        help='GOAC solver: random reproduces the paper, sa/mc/ga optimize further',
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=200000,
        help='optimizer steps (MC/SA sweeps, or GA generations); ignored by --solver random',
    )
    parser.add_argument(
        '--starts',
        type=int,
        default=20,
        help='random starting points for the optimizing solvers',
    )
    parser.add_argument(
        '--disorder-partners',
        action='store_true',
        help='also permute the elements sharing sites with --specie (Y in Li3YCl3Br3) '
        'over those positions',
    )
    parser.add_argument(
        '--extra-interstitials',
        action='store_true',
        help='add the empty voids of the anion sublattice to the mobile positions, so that '
        'x can go beyond the cap set by the sites the CIF lists',
    )
    parser.add_argument(
        '--void-radius',
        type=float,
        default=2.3,
        help='minimum distance in Å from an anion for a void to count as an interstitial '
        'site (only with --extra-interstitials)',
    )
    parser.add_argument(
        '--void-separation',
        type=float,
        default=2.2,
        help='minimum distance in Å from any other cation position for a void to be added '
        '(only with --extra-interstitials)',
    )
    parser.add_argument(
        '--n-best',
        type=int,
        default=1,
        help='how many lowest-energy structures to write as CIFs per composition',
    )
    parser.add_argument(
        '--tol',
        type=float,
        default=DEFAULT_TOL,
        help='energies within this many eV count as the same solution when collecting '
        '--n-best, so degenerate orderings are not written twice (-1 keeps them all); '
        'only the sa/mc solvers filter',
    )
    output = parser.add_argument(
        '--output',
        type=Path,
        help='directory for the CIFs and sweep.png',
    )
    # It has no real default: a fresh temp directory is made at run time instead.
    output.default_text = 'TEMPDIR'  # type: ignore[attr-defined]

    parser.add_argument('--x-min', type=float, default=0.0, help='lowest x to sample')
    parser.add_argument(
        '--x-max',
        type=float,
        help='highest x to sample (default: every mobile position filled, more with '
        '--extra-interstitials)',
    )
    parser.add_argument('--x-step', type=float, default=0.5, help='spacing in x')
    parser.add_argument(
        '--scatter-max',
        type=int,
        default=2000,
        help='most sampled configurations to scatter per composition',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help="don't report progress, and swallow GOAC's own solver output; the CIFs and "
        'sweep.png are still written',
    )
    parser.add_argument(
        '--progress',
        action='store_true',
        help='draw a progress bar over the compositions on stderr; implies swallowing '
        "GOAC's own solver output, which would otherwise bury the bar",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    # The parser's dest names match GOACSweep's constructor one-for-one.
    GOACSweep(**vars(args)).run()


if __name__ == '__main__':
    main()
