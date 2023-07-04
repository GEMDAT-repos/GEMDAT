import sys
from pathlib import Path

from streamlit.web.cli import main


def gemdash():
    run_file = str(Path(__file__).resolve().parent / 'run.py')
    sys.argv = ['streamlit', 'run', run_file] + sys.argv[1:]
    main()
