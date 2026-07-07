# Frequently asked questions

This page collects answers to questions that come up regularly, mostly distilled
from the [issue tracker](https://github.com/GEMDAT-repos/GEMDAT/issues). If your
question is not answered here, feel free to
[open an issue](https://github.com/GEMDAT-repos/GEMDAT/issues/new).

## Loading simulation data

### Does GEMDAT support CP2K trajectories or plain `.xyz` files?

There is no dedicated CP2K reader, but a CP2K trajectory in standard `.xyz`
format can be read through the LAMMPS loader,
[`Trajectory.from_lammps`][gemdat.trajectory.Trajectory.from_lammps], which
accepts the `xyz` coordinate format:

```python
from gemdat import Trajectory

trajectory = Trajectory.from_lammps(
    coords_file='trajectory.xyz',
    data_file='lattice.data',
    coords_format='xyz',
    temperature=700,
    time_step=0.001,  # in ps
)
```

An `.xyz` file only contains coordinates, not the simulation box, so you also
have to supply the lattice through `data_file` (a LAMMPS data file describing the
box). If the coordinate file labels species by number rather than by element,
use the `type_mapping` argument to map them, e.g. `type_mapping={'1': 'Li', '2':
'S'}`.

See [issue #368](https://github.com/GEMDAT-repos/GEMDAT/issues/368) for the
original discussion.

## Sites, densities and interstitials

### How do I handle interstitials or atoms that do not align with my sites?

If the diffusing atoms occupy interstitial positions that are not part of your
site definitions, jumps to those positions cannot be detected and simply
enlarging the site radius will not capture them. Instead, let GEMDAT derive the
sites from the trajectory density itself:

```python
volume = trajectory.to_volume()
structure = volume.to_structure()
```

[`Volume.to_structure`][gemdat.volume.Volume.to_structure] detects **all** density
peaks — including interstitials — and returns a
[`Structure`][pymatgen.core.structure.Structure] with sites at those positions.
Alternatively, [`ShapeAnalyzer.optimize_sites`][gemdat.shape.ShapeAnalyzer.optimize_sites]
can shift your existing sites to where the atoms actually reside.

Once you have a corrected structure, pass it to
[`transitions_between_sites`][gemdat.trajectory.Trajectory.transitions_between_sites]
with a per-site `site_radius` dictionary keyed by site label, so that
interstitial sites can be given a different radius where needed:

```python
transitions = trajectory.transitions_between_sites(
    sites=structure,
    floating_specie='Li',
    site_radius={'Li1': 1.0, 'Li2': 1.5},
)
```

See [issue #371](https://github.com/GEMDAT-repos/GEMDAT/issues/371).

### My sites are rotated or distorted compared to the ideal structure — what should I do?

Different layers of a structure can distort differently during the MD, and the
diffusion probability density faithfully reports what the simulation produces —
so an apparent rotation of the occupied sites relative to the ideal crystal is
usually a real physical effect rather than an artifact.

To align your site definitions with the centres of the occupied regions, use
[`ShapeAnalyzer.optimize_sites`][gemdat.shape.ShapeAnalyzer.optimize_sites],
which by default shifts each site to its
[`ShapeData.centroid`][gemdat.shape.ShapeData.centroid] and returns a new
`ShapeAnalyzer` with the corrected positions. For manual adjustments with custom
displacement vectors, use
[`ShapeAnalyzer.shift_sites`][gemdat.shape.ShapeAnalyzer.shift_sites].

See [issue #370](https://github.com/GEMDAT-repos/GEMDAT/issues/370).

## Metrics

### What exactly does GEMDAT's "vibration amplitude" measure?

The vibration amplitude reported by
[`TrajectoryMetrics.vibration_amplitude`][gemdat.metrics.TrajectoryMetrics.vibration_amplitude]
is a **global empirical displacement measure referenced to each ion's initial
position**, not a local per-site rattling amplitude.

Concretely, GEMDAT works from the scalar distance of each ion to its base
position — the first frame, `coords[0]` — using the full trajectory without any
time windowing. Following Eq. 5 of the
[GEMDAT paper](https://doi.org/10.1038/s41524-026-02133-7), this displacement
signal is split into segments where the discrete-time derivative keeps a constant
sign, and the amplitude of each segment is `A = r_i(t_b) − r_i(t_a)`. The
resulting amplitudes are assumed to be Gaussian-distributed, and
`vibration_amplitude()` reports their standard deviation as the characteristic
amplitude.

A consequence worth keeping in mind: for an ion that has migrated several Å from
its starting point, the measure mostly tracks the component of local vibration
*parallel* to the displacement vector from the initial position, and is weakly
sensitive to perpendicular motion.

See [issue #397](https://github.com/GEMDAT-repos/GEMDAT/issues/397).
