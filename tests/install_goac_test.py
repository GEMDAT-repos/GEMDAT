"""Test that the GOAC installation helper script works end-to-end."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / 'tools' / 'install_goac.py'


@pytest.mark.skipif(
    not (
        subprocess.run(['git', '--version'], capture_output=True).returncode == 0
        and subprocess.run(['gfortran', '--version'], capture_output=True).returncode == 0
    ),
    reason='Requires git and gfortran on PATH',
)
class TestGOACInstallation:
    """Tests that the GOAC install script runs end-to-end on a suitable
    machine."""

    def test_install_goac(self):
        """Run the install script and verify GOAC can be imported
        afterwards."""
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
