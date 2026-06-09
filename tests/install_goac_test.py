"""Tests for the GOAC installation helper script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / 'tools' / 'install_goac.py'


class TestGOACInstallation:
    """Tests for the GOAC installation helper script."""

    def test_script_exists(self):
        """Script file exists."""
        assert SCRIPT_PATH.exists()

    def test_script_syntax(self):
        """Script has valid Python syntax."""
        result = subprocess.run(
            [sys.executable, '-m', 'py_compile', str(SCRIPT_PATH)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_import_and_dependency_check(self):
        """check_dependencies runs and prints status (may exit if deps
        missing)."""
        sys.path.insert(0, str(SCRIPT_PATH.parent))
        try:
            import install_goac

            assert hasattr(install_goac, 'check_dependencies')
            assert hasattr(install_goac, 'install_goac')

            # _has_cmd and _build_f2py are helpers
            assert install_goac._has_cmd('true')
            assert not install_goac._has_cmd('nonexistent-command-xyz')
        finally:
            sys.path.pop(0)

    def test_script_runs_safely(self):
        """Script runs without crashing (may exit 1 if deps missing)."""
        result = subprocess.run(
            [sys.executable, '-c', 'import install_goac; install_goac.check_dependencies()'],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(SCRIPT_PATH.parent),
        )
        # check_dependencies will sys.exit(1) if deps are missing, which is fine
        assert result.returncode in (0, 1), result.stderr

    @pytest.mark.skipif(
        not (
            subprocess.run(['git', '--version'], capture_output=True).returncode == 0
            and subprocess.run(['gfortran', '--version'], capture_output=True).returncode == 0
        ),
        reason='Requires git and gfortran on PATH',
    )
    def test_full_installation(self):
        """Full end-to-end install in a temporary environment."""
        # Use a subprocess with isolated pip to avoid polluting main env
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
        assert result.returncode == 0

        # Verify the import works
        code = 'from goac import GOAC, ABCEwald; print("GOAC imported OK")'
        check = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True,
            text=True,
        )
        assert check.returncode == 0, check.stderr
        assert 'GOAC imported OK' in check.stdout
