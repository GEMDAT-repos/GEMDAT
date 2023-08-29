This page shows you how to get started using Gemdat in your own Python scripts or in a Jupyter notebook.

## Loading simulation data

The first step is to load your simulation data. These are represented in gemdat by a trajectory. A trajectory is essentially a list of coordinates for all species over time. To load a trajectory:

```python
from gemdat import Trajectory

trajectory = Trajectory.from_vasprun('./path/to/vasprun.xml')
```

For now, select the trajectory for the diffusing species, Lithium, by its symbol:

```python
diff_trajectory = trajectory.filter('Li')
```

## Plotting trajectory properties

You can do this by importing the `plots` submodule. In the example below, we are plotting the vibration of the diffusing element by passing its trajectory.

```python
from gemdat import plots

plots.plot_vibrational_amplitudes(diff_trajectory)
```

See the [plots page](./plotting.md#trajectory-and-displacements-plots) for more information about which plots for trajectories are available.

## Analyzing jumps and transitions

To find out how Lithium jumps from site to site, we must know what the sites are. Gemdat contains a [small internal database](https://github.com/GEMDAT-repos/GEMDAT/tree/main/src/gemdat/data) with available structures.

Alternatively, you can load the sites from 1. [a cif file][gemdat.io.read_cif], 2. [a density or volume][gemdat.volume.volume_to_structure] or 3. construct your own [Structure][pymatgen.core.structure.Structure].

```python
from gemdat import load_known_material

structure = load_known_material('argyrodite', supercell=(2, 1, 1))
```

The [gemdat.SitesData][] class is responsible for calculating the different jumps properties. We pass the `structure` with the jump sites, the trajectory containing both the host and diffusing species, and specify what the diffusing element is.

```python
from gemdat import SitesData

sites = SitesData(
   structure=structure,
   trajectory=trajectory,
   floating_specie='Li',
)
```

See the documentation for [`gemdat.SitesData`][] to find out which properties are available.

## Plotting jumps and transition properties

Gemdat contains [several functions for plotting jumps and transitions](./plotting.md#jumps-and-transition-plots). These take the [gemdat.SitesData][] object we just constructed as input.

For example, to visualize all jumps in 3d:

```
plots.plot_jumps_3d(sites)
```

## Radial distribution functions

This function calculates the [radial distribution function](https://en.wikipedia.org/wiki/Radial_distribution_function) (RDF) from the `floating_specie` on [gemdat.SitesData][],

```python
from gemdat.rdf import radial_distribution

rdf_data = radial_distribution(sites=sites)
```

This returns a dictionary with all possible transitions between the jump sites. You can use the [plotting module](./plotting.md#radial-distribution-plots) to visualize the RDFs:

```python
for rdfs in rdf_data.values():
   plots.radial_distribution(rdfs)
```
