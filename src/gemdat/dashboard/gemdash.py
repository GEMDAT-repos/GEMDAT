from __future__ import annotations

import sys
from importlib.resources import files
from pathlib import Path

from streamlit.web.cli import main

dashboard_directory = Path(files('gemdat') / 'dashboard')  # type: ignore


def gemdash():
    run_file = str(dashboard_directory / 'run.py')

    sys.argv = [
        *('streamlit', 'run', run_file),
        *('--theme.base', 'light'),
        *('--theme.primaryColor', 'e03c31'),
        *('--theme.secondaryBackgroundColor', 'bcd9ec'),
        *('--browser.gatherUsageStats', 'false'),
        *sys.argv[1:],
    ]

    main()
