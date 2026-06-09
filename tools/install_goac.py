#!/usr/bin/env python3
"""Install GOAC (Global Optimization of Atomistic Configurations by Coulomb).

GOAC requires compilation of Fortran code via f2py. System prerequisites:
  - gfortran, meson >= 1.6.0, ninja >= 1.11.1.2, OpenMP, pkg-config

Usage:
    python tools/install_goac.py
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
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'numpy.f2py',
            '-c',
            source,
            '-m',
            module,
            '--backend',
            'meson',
            '--dep',
            'openmp',
        ],
        cwd=work_dir,
        capture_output=True,
        text=True,
    )
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
            'from GOAC import *\n'
            'from GOAC import energy, monte_carlo, ga, remc, rega, greedy, '
            'local_minimizer, branch_n_bound, occupied, valid_solutions, '
            'solution_unique, random_samples\n'
            'from .IterationProblem import Iteration_Problem\n'
            'from .GreedySolver import Greedy_Solver\n'
            'from .RandomSolver import Random_Solver\n'
            'from .Solver import Solver\n'
            'import ABCEwald\n'
            '__version__ = "2024.1.0"\n'
        )

        (pkg / 'pyproject.toml').write_text(
            '[build-system]\n'
            'requires = ["setuptools"]\n'
            'build-backend = "setuptools.build_meta"\n'
            '\n'
            '[project]\n'
            'name = "goac"\n'
            'version = "2024.1.0"\n'
            'description = "Global Optimization of Atomistic Configurations by Coulomb"\n'
        )

        print('  Installing GOAC package...')
        subprocess.run([sys.executable, '-m', 'pip', 'install', str(pkg)], check=True)
    print('\n✓ GOAC installed successfully!')


def main() -> None:
    print('=' * 50, 'GOAC Installation', sep='\n')
    check_dependencies()
    install_goac()


if __name__ == '__main__':
    main()
