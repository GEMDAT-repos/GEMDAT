[![Documentation Status](https://readthedocs.org/projects/gemdat/badge/?version=latest)](https://gemdat.readthedocs.io/en/latest/?badge=latest)
[![Tests for GEMDAT](https://github.com/GEMDAT-repos/GEMDAT/actions/workflows/tests.yaml/badge.svg)](https://github.com/GEMDAT-repos/GEMDAT/actions/workflows/tests.yaml)
![Coverage](https://gist.githubusercontent.com/stefsmeets/b599ff4ccf4a6d201a984502f049da73/raw/covbadge.svg)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/gemdat)](https://pypi.org/project/gemdat/)
[![PyPI](https://img.shields.io/pypi/v/gemdat.svg?style=flat)](https://pypi.org/project/gemdat/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8401669.svg)](https://doi.org/10.5281/zenodo.8401669)

![GEMDAT banner](https://raw.githubusercontent.com/GEMDAT-repos/GEMDAT/main/src/gemdat/data/logo_on_white.png#gh-light-mode-only)
![GEMDAT banner](https://raw.githubusercontent.com/GEMDAT-repos/GEMDAT/main/src/gemdat/data/logo_on_black.png#gh-dark-mode-only)

# GEMDAT

Gemdat is a Python library for the analysis of diffusion in solid-state electrolytes from Molecular Dynamics simulations. Gemdat is built on top of [Pymatgen](https://pymatgen.org/), making it straightforward to integrate it into your Pymatgen-based workflows.

With Gemdat, you can:

- Explore your MD simulation via an easy-to-use Python API
- Load and analyze trajectories from VASP and LAMMPS simulation data
- Find jumps and transitions between sites
- Effortlessly calculate tracer and jump diffusivity
- Characterize and visualize diffusion pathways
- Plot radial distribution functions

To install:

```console
pip install gemdat
```

The source code is available from [Github](https://github.com/GEMDAT-repos/GEMDAT).

Suggestions, improvements, and edits are most welcome.

## Usage

The following snippet to analyze the diffusion trajectory from VASP data.

```python
from gemdat import Trajectory

trajectory = Trajectory.from_vasprun('../example/vasprun.xml')

trajectory.plot_displacement_per_element()

diff_trajectory = trajectory.filter('Li')

diff_trajectory.plot_displacement_per_atom()
diff_trajectory.plot_displacement_histogram()
diff_trajectory.plot_frequency_vs_occurence()
diff_trajectory.plot_vibrational_amplitudes()
```

Characterize transitions and jumps between sites:

```python
from gemdat.io import load_known_material

sites = load_known_material('argyrodite', supercell=(2, 1, 1))

transitions = trajectory.transitions_between_sites(
    sites=sites,
    floating_specie='Li',
)

jumps = transitions.jumps()

jumps.plot_jumps_vs_distance()
jumps.plot_jumps_vs_time()
jumps.plot_collective_jumps()
jumps.plot_jumps_3d()

jumps.jump_diffusivity(dimensions=3)
```

To calculate different metrics, such as tracer diffusivity:

```python
from gemdat import TrajectoryMetrics

metrics = TrajectoryMetrics(diff_trajectory)

metrics.tracer_diffusivity(dimensions=3)
metrics.haven_ratio(dimensions=3)
metrics.tracer_conductivity(dimensions=3)
metrics.particle_density()
metrics.vibration_amplitude()
```

## Development

Check out our [Contributing Guidelines](CONTRIBUTING.md#Getting-started-with-development) to get started with development.

## How to Cite

- Victor Azizi, Stef Smeets, Anastasiia K. Lavrinenko and Simone Ciarella. GEMDAT (Version 1.6.0) [Computer software]. https://github.com/GEMDAT-repos/GEMDAT, doi: [10.5281/zenodo.8401669](https://dx.doi.org/10.5281/zenodo.8401669)

## Credits

The code in this repository is based on [Matlab code to analyse Molecular Dynamics simulations](https://bitbucket.org/niekdeklerk/md-analysis-with-matlab/src/master/).

For background information on how some of the properties are calculated, check out the accompanying paper:

- Niek J.J. de Klerk, Eveline van der Maas and Marnix Wagemaker, ACS Applied Energy Materials, (2018), doi: [10.1021/acsaem.8b00457](https://doi.org/10.1021/acsaem.8b00457)
