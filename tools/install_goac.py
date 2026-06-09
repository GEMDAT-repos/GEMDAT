#!/usr/bin/env python3
"""Install GOAC (Global Optimization of Atomistic Configurations by Coulomb).

GOAC requires compilation of Fortran code via f2py. System prerequisites:
  - gfortran, meson >= 1.6.0, ninja >= 1.11.1.2, OpenMP, pkg-config

Usage:
    python tools/install_goac.py

Installation strategy
---------------------
GOAC is not available on PyPI and its original build system relies on the classic
``numpy.distutils`` (deprecated). We handle installation as follows:

1. **Clone** the GOAC repository from the IFF GitLab.
2. **Build** the two Fortran extension modules (``GOAC`` and ``ABCEwald``) via
   ``numpy.f2py`` with the ``meson`` backend and ``openmp`` dependency.
3. **Rewrite imports** in the Python source files so that relative imports
   (``from .Solver import``) are used instead of top-level ones, making the
   code work as a proper package.
4. **Assemble** a minimal package structure with an ``__init__.py`` that
   re-exports all public symbols.
5. **Install** directly into the active environment's ``site-packages/goac/``
   by copying the compiled ``.so`` files and rewritten ``.py`` files.

This avoids depending on a full ``setup.py`` / ``pyproject.toml`` for GOAC,
and mirrors what a ``pip install`` of a versioned release would do, but with
a fresh build of the Fortran sources against the local toolchain.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _has_cmd(cmd: str) -> bool:
    """Return True if *cmd* runs successfully."""
    try:
        return subprocess.run(cmd.split(), capture_output=True, check=False).returncode == 0
    except FileNotFoundError:
        return False


def check_dependencies() -> None:
    """Print dependency status and exit if anything is missing."""
    ok = True
    for name, cmd in (
        ('gfortran', 'gfortran --version'),
        ('meson', 'meson --version'),
        ('ninja', 'ninja --version'),
    ):
        if _has_cmd(cmd):
            print(f'  ✓ {name}')
        else:
            print(
                f'  ✗ {name}  (install: apt-get install gfortran meson ninja-build libomp-dev)'
            )
            ok = False

    if not _has_cmd('pkg-config --exists omp'):
        print('  ⚠ OpenMP not detected via pkg-config (may still work)')

    for mod, pkg in (('mesonpy', 'meson-python'), ('numpy', 'numpy'), ('pymatgen', 'pymatgen')):
        try:
            __import__(mod)
            print(f'  ✓ {pkg}')
        except ImportError:
            print(f'  ✗ {pkg}')
            ok = False

    if not ok:
        print('\nMissing dependencies — install them and retry.')
        sys.exit(1)


def _build_f2py(work_dir: Path, source: str, module: str) -> Path:
    """Build one f2py extension and return the .so path."""
    print(f'  Building {module}...')
    base_cmd = [
        sys.executable,
        '-m',
        'numpy.f2py',
        '-c',
        source,
        '-m',
        module,
        '--backend',
        'meson',
    ]
    # Try with OpenMP first, fall back to without if meson can't find it
    for extra in (['--dep', 'openmp'], []):
        result = subprocess.run(
            base_cmd + extra,
            cwd=work_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            break
        if extra:
            print('  OpenMP build failed, retrying without OpenMP...')
    if result.returncode != 0:
        sys.exit(f'f2py build failed for {module}:\n{result.stderr}')
    return next(work_dir.glob(f'{module}*.so'))


def install_goac() -> None:
    """Clone GOAC, build Fortran extensions, and pip-install the package."""
    with tempfile.TemporaryDirectory(prefix='goac-') as tmp_str:
        tmp = Path(tmp_str)
        print('  Cloning GOAC repository...')
        subprocess.run(
            ['git', 'clone', 'https://iffgit.fz-juelich.de/k.koester/goac.git', str(tmp)],
            check=True,
            capture_output=True,
        )

        src = tmp / 'GOAC'
        so1 = _build_f2py(src, 'GOAC.f90', 'GOAC')
        so2 = _build_f2py(src, 'ABCEwald.f90', 'ABCEwald')

        # Assemble a pip-installable package
        pkg = tmp / 'goac_pkg'
        pkg.mkdir()
        for so in (so1, so2):
            shutil.copy(so, pkg)

        for fn in (
            'IterationProblem.py',
            'GreedySolver.py',
            'RandomSolver.py',
            'Solver.py',
            '__main__.py',
        ):
            text = (src / fn).read_text()
            for old, new in (
                ('from Solver import', 'from .Solver import'),
                ('from IterationProblem import', 'from .IterationProblem import'),
                ('from ABCEwald import', 'from .ABCEwald import'),
            ):
                text = text.replace(old, new)
            (pkg / fn).write_text(text)

        (pkg / '__init__.py').write_text(
            'from .GOAC import *\n'
            'from .GOAC import energy, monte_carlo, ga, remc, rega, greedy, '
            'local_minimizer, branch_n_bound, occupied, valid_solutions, '
            'solution_unique, random_samples\n'
            'from .IterationProblem import Iteration_Problem\n'
            'from .GreedySolver import Greedy_Solver\n'
            'from .RandomSolver import Random_Solver\n'
            'from .Solver import Solver\n'
            'from . import ABCEwald\n'
            '__version__ = "2024.1.0"\n'
        )

        print('  Installing GOAC package...')

        # Install the .py files and .so extensions directly into site-packages
        site_packages = Path(
            subprocess.run(
                [sys.executable, '-c', 'import site; print(site.getsitepackages()[0])'],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )

        pkg_dst = site_packages / 'goac'
        pkg_dst.mkdir(exist_ok=True)

        for so in (so1, so2):
            shutil.copy(so, pkg_dst)

        for fn in (
            '__init__.py',
            'IterationProblem.py',
            'GreedySolver.py',
            'RandomSolver.py',
            'Solver.py',
            '__main__.py',
        ):
            shutil.copy(pkg / fn, pkg_dst)

    print('\n\u2713 GOAC installed successfully!')


def main() -> None:
    print('=' * 50, 'GOAC Installation', sep='\n')
    check_dependencies()
    install_goac()


if __name__ == '__main__':
    main()
