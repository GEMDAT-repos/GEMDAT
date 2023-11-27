from __future__ import annotations

import argparse
import sys
from importlib.resources import files
from pathlib import Path

from streamlit.web.cli import main

dashboard_directory = Path(files('gemdat') / 'dashboard')  # type: ignore


def gemdash():
    parser = argparse.ArgumentParser(
        prog='gemdash',
        add_help=False,
        description='Streamlit dashboard for easily visualizing gemdat data')
    parser.add_argument('--file',
                        '-f',
                        nargs='?',
                        default='vasprun.xml',
                        help='File to load in gemdash')
    parser.add_argument('--help',
                        action='count',
                        default=0,
                        help='specify twice to print streamlit help')

    arguments, unknown = parser.parse_known_args(sys.argv[1:])
    if arguments.help == 1:
        parser.print_help()
        return
    if arguments.help == 2:
        unknown.append('--help')

    run_file = str(dashboard_directory / 'run.py')

    sys.argv = [
        *('streamlit', 'run', run_file),
        *('--theme.base', 'light'),
        *('--theme.primaryColor', 'e03c31'),
        *('--theme.secondaryBackgroundColor', 'bcd9ec'),
        *('--browser.gatherUsageStats', 'false'),
        *unknown,
        '--',
        *('--file', arguments.file),
    ]

    main()
