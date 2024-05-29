from __future__ import annotations


def test_matplotlib_imports():
    from gemdat.plots import matplotlib

    assert matplotlib.__all__


def test_plotly_imports():
    from gemdat.plots import plotly

    assert plotly.__all__
