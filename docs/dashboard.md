GEMDAT includes a dashboard to interactively select your input data and experimental parameters. This makes it straightforward to explore your data without any coding experience required.

The dashboard is available from: https://github.com/GEMDAT-repos/gemdat-dashboard

To use the dashboard, first make sure you install all the dependencies:

```
pip install -r requirements.txt
```

The dashboard is available via the command:

```
python gemdash.py
```

This will start a server with a URL you can paste in your browser to visit the dashboard.

!!! note

    The dashboard runs in the web browser and needs a somewhat recent version of [firefox or one of the other supported browsers here](https://docs.streamlit.io/knowledge-base/using-streamlit/supported-browsers).

To pre-set your data, you can set the `VASP_XML` environment variable (useful for testing):

```
VASP_XML=./path/to/vasprun.xml gemdash
```

It is also possible to specify the file that you want to load via the commandline:
```
python gemdash.py --file newrun/vasprun.xml
```

The options of gemdat can be listed with `python gemdash.py --help`:

```
usage: gemdash [--file [FILE]] [--help]

Streamlit dashboard for easily visualizing gemdat data

options:
  --file [FILE], -f [FILE]
                        File to load in gemdash
  --help                specify twice to print streamlit help
```
