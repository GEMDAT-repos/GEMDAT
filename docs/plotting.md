# Plots

Gemdat contains several built-in plots for visualizing trajectories, jumps, transitions, and radial distribution functions.

These are collected in the `plots` module. The intended usage is that you import `gemdat.plots` like this:

```python
from gemdat import plots

plots.displacement_per_element(trajectory)
plots.jumps_vs_distance(trajectory, sites)
plots.radial_distribution(rdfs)
```

All plotting functions take a [gemdat.Trajectory][], [gemdat.Jumps][], [gemdat.Transitions][], [gemdat.rdf.RDFData][] or a combination as input. In addition, for some plots you have a few parameters to tune the output.

All available plots are documented in the [gemdat.plots API reference](api/gemdat_plots.md). Some highlights are listed below.

## Trajectory and displacements plots

- [gemdat.plots.displacement_per_atom][]
- [gemdat.plots.displacement_per_element][]
- [gemdat.plots.displacement_histogram][]

## Simulation metrics plots

- [gemdat.plots.frequency_vs_occurence][]
- [gemdat.plots.vibrational_amplitudes][]

## Jumps and transition plots

- [gemdat.plots.jumps_vs_distance][]
- [gemdat.plots.jumps_vs_time][]
- [gemdat.plots.collective_jumps][]
- [gemdat.plots.jumps_3d][]
- [gemdat.plots.jumps_3d_animation][]

## Radial distribution plots

- [gemdat.plots.radial_distribution][]
