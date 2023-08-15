[![Documentation Status](https://readthedocs.org/projects/gemdat/badge/?version=latest)](https://gemdat.readthedocs.io/en/latest/?badge=latest)
[![Tests for GEMDAT](https://github.com/GEMDAT-repos/GEMDAT/actions/workflows/tests.yaml/badge.svg)](https://github.com/GEMDAT-repos/GEMDAT/actions/workflows/tests.yaml)
![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/v1kko/330d6e711de3420b7503d54756dc011c/raw/covbadge.json)
<!-- [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/XXX)](https://pypi.org/project/XXX/) -->
<!-- [![PyPI](https://img.shields.io/pypi/v/XXX.svg?style=flat)](https://pypi.org/project/XXX/) -->

![GEMDAT banner](https://raw.githubusercontent.com/GEMDAT-repos/GEMDAT/main/src/data/logo.png)

# GEMDAT

This repository contains Python code to analyse Molecular Dynamics simulations.

The code in this repository is based on:
https://bitbucket.org/niekdeklerk/md-analysis-with-matlab/src/master/

## Installation

To install:

```console
pip install .[develop]
```

The source code is available from [Github](https://github.com/GEMDAT-repos/GEMDAT).

Suggestions, improvements, and edits are most welcome.

## Usage

The following snippet can be used to test the code using VASP data.

```python
from gemdat import Trajectory, plots
from gemdat.calculate.extras import calculate_all

trajectory = Trajectory.from_vasprun(Path('../example/vasprun.xml'))

diff_trajectory = trajectory.filter('Li')

plots.plot_displacement_per_element(trajectory)
plots.plot_displacement_per_site(diff_trajectory)
plots.plot_displacement_histogram(diff_trajectory)
plots.plot_frequency_vs_occurence(trajectory)
plots.plot_vibrational_amplitudes(trajectory)

structure = load_known_material('argyrodite', supercell=(2, 1, 1))

sites = SitesData(structure)
sites.calculate_all(trajectory=trajectory,
                    diffusing_element='Li')

plots.plot_jumps_vs_distance(trajectory, sites)
plots.plot_jumps_vs_time(trajectory, sites)
plots.plot_collective_jumps(trajectory, sites)
plots.plot_jumps_3d(trajectory, sites)
```

Or, one function to do everything:

```python
from gemdat.legacy import analyse_md

trajectory, sites, extras = analyse_md(
   '/data/vasprun.xml',
   diff_elem='Li',
   supercell=(2, 1, 1),
   material='argyrodite',
)
```

## Development

Check out our [Contributing Guidelines](CONTRIBUTING.md#Getting-started-with-development) to get started with development.

## References

- Niek J.J. de Klerk, Eveline van der Maas and Marnix Wagemaker, ACS Applied Energy Materials, (2018), doi: [10.1021/acsaem.8b00457](https://doi.org/10.1021/acsaem.8b00457)
