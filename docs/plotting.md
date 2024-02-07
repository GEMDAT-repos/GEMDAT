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


## Trajectory and displacements plots

::: gemdat.plots
    options:
      show_root_toc_entry: false
      heading_level: 3
      members:
        - displacement_per_site
        - displacement_per_element
        - displacement_histogram


## Simulation metrics plots

::: gemdat.plots
    options:
      show_root_toc_entry: false
      heading_level: 3
      members:
        - frequency_vs_occurence
        - vibrational_amplitudes

## Jumps and transition plots

::: gemdat.plots
    options:
      show_root_toc_entry: false
      heading_level: 3
      members:
        - jumps_vs_distance
        - jumps_vs_time
        - collective_jumps
        - jumps_3d
        - jumps_3d_animation

## Radial distribution plots

::: gemdat.plots
    options:
      show_root_toc_entry: false
      heading_level: 3
      members:
        - radial_distribution
