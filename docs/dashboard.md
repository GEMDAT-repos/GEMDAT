Gemdash includes a dashboard to interactively select your input data and experimental parameters. This makes it straightforward to explore your data without any coding experience required.

To use the dashboard, first make sure you install all the dependencies:

```
pip install gemdat[dashboard]
```

The dashboard is available via the command

```
gemdash
```

To pre-set your data, you can set the `VASP_XML` environment variable (useful for testing):

```
VASP_XML=./path/to/vasprun.xml gemdash
```

It is also possible to specify the file that you want to load via the commandline:
```
gemdash newrun/vasprun.xml
```

The options of gemdat can be listed with `gemdash --help`:
```
usage: gemdash [--help] [filename]

Streamlit dashboard for easily visualizing gemdat data

positional arguments:
  filename  File to load in gemdash

options:
  --help    specify twice to print streamlit help
```

!!! note

    The dashboard runs in the web browser and needs a somewhat recent version of [firefox or one of the other supported browsers here](https://docs.streamlit.io/knowledge-base/using-streamlit/supported-browsers).
