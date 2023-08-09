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
from gemdat import Trajectory, plot_all, plot
from gemdat.calculate.extras import calculate_all

trajectory = Trajectory.from_vasprun(Path('../example/vasprun.xml'))
extras = calculate_all(diffusing_element='Li')

structure = load_known_material('argyrodite', supercell=(2, 1, 1))

sites = SitesData(structure)
sites.calculate_all(trajectory=trajectory,
                    extras=extras)

plot_all(trajectory=trajectory,
         sites=sites,
         **vars(extras),
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
