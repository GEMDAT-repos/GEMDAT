[![Documentation Status](https://readthedocs.org/projects/gemdat/badge/?version=latest)](https://gemdat.readthedocs.io/en/latest/?badge=latest)
[![Tests for GEMDAT](https://github.com/GEMDAT-repos/GEMDAT/actions/workflows/tests.yaml/badge.svg)](https://github.com/GEMDAT-repos/GEMDAT/actions/workflows/tests.yaml)
![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/v1kko/330d6e711de3420b7503d54756dc011c/raw/covbadge.json)
<!-- [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/XXX)](https://pypi.org/project/XXX/) -->
<!-- [![PyPI](https://img.shields.io/pypi/v/XXX.svg?style=flat)](https://pypi.org/project/XXX/) -->

![GEMDAT banner](https://raw.githubusercontent.com/GEMDAT-repos/GEMDAT/main/docs/logo.png)

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

The following snippet can be used to test the code (provided that you have some VASP data or intermediate `.mat` files.

```python
from pathlib import Path

from gemdat import Data, plot_all, plot
from gemdat.calculate import calculate_displacements, calculate_speed, calculate_diff_displacements, calculate_amplitudes, calculate_vibration_amplitude, calculate_attempt_freq, calculate_attempt_freq_std

data = Data.from_vasprun(Path('../example/vasprun.xml'), cache=Path('cache'))
displacements = calculate_displacements(data.trajectory_coords, data.lattice)
diff_displacements = calculate_diff_displacements(displacements=displacements, species=data.species)
speed = calculate_speed(diff_displacements)
freq = calculate_attempt_freq(diff_displacements)
freq_std = calculate_attempt_freq_std(diff_displacements)
amplitudes = calculate_amplitudes(speed)
vibration_amplitude = calculate_vibration_amplitude(speed)

plot(data = data, plots = ['plot_frequency_vs_occurence'])
# or

plot_all(data = data, displacements=displacements, speed=speed, amplitudes=amplitudes, vibration_amplitude=vibration_amplitude, freq=freq, freq_std=freq_std)
```

**not yet available:**
```python
import gemdat
gemdat.analyse_md('<path to data>', 'Li', 'argyrodite')
```

## Development

Check out our [Contributing Guidelines](CONTRIBUTING.md#Getting-started-with-development) to get started with development.

## References

- Niek J.J. de Klerk, Eveline van der Maas and Marnix Wagemaker, ACS Applied Energy Materials, (2018), doi: [10.1021/acsaem.8b00457](https://doi.org/10.1021/acsaem.8b00457)
