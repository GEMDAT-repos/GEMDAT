from __future__ import annotations


def test_matplotlib_imports():
    from gemdat.plots import matplotlib

    assert matplotlib.__all__


def test_plotly_imports():
    from gemdat.plots import plotly

    assert plotly.__all__


def test_get_axes():
    import matplotlib.pyplot as plt
    import pytest

    from gemdat.plots.matplotlib._axes import get_axes

    fig, ax = get_axes(None)
    assert ax.figure is fig

    _, user_ax = plt.subplots()
    assert get_axes(user_ax) == (None, user_ax)

    ax3d = plt.figure().add_subplot(projection='3d')
    assert get_axes(ax3d, projection='3d') == (None, ax3d)
    assert get_axes(None, projection='3d')[1].name == '3d'

    with pytest.raises(TypeError, match='3D axes'):
        get_axes(user_ax, projection='3d')

    plt.close('all')
