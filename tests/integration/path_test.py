from __future__ import annotations

from math import isclose

import numpy as np
import pytest
from numpy.testing import assert_allclose

from gemdat.io import load_known_material
from gemdat.path import optimal_n_paths, optimal_path


@pytest.vaspxml_available  # type: ignore
def test_fractional_coordinates(vasp_path_vol, vasp_path):
    frac_sites = vasp_path.frac_sites()

    assert isclose(frac_sites[-1][0], 0.4107142857142857)
    assert isclose(frac_sites[19][1], 0.6785714285714286)
    assert isclose(frac_sites[10][2], 0.4642857142857143)

    assert np.all(frac_sites <= 1)
    assert np.all(frac_sites >= 0)


TEST_DATA = (
    # start, stop, method, expected
    ((10, 4, 13), (21, 3, 10), 'simple', 6.3023933200094655),
    ((10, 4, 13), (21, 3, 10), 'dijkstra', 5.210130709736149),
    ((7, 9, 2), (20, 4, 2), 'bellman-ford', 5.5363545176821),
    ((18, 7, 12), (25, 3, 13), 'dijkstra-exp', 5.029753032964493),
    ((3, 3, 6), (0, 9, 4), 'minmax-energy', 2.708267913357463),
)


@pytest.vaspxml_available  # type: ignore
@pytest.mark.parametrize('start,stop,method,expected', TEST_DATA)
def test_optimal_path(vasp_F_graph, start, stop, method, expected):
    path = optimal_path(vasp_F_graph, start=start, stop=stop, method=method)
    assert isclose(path.total_energy, expected)


@pytest.vaspxml_available  # type: ignore
def test_optimal_percolating_path(vasp_path):
    assert isclose(vasp_path.total_energy, 11.488013690080908)
    assert vasp_path.start_site == (11, 9, 6)


@pytest.vaspxml_available  # type: ignore
def test_path_length(vasp_path_vol, vasp_path):
    structure = load_known_material('argyrodite')
    length = vasp_path.total_length(structure.lattice)
    assert length == 20.10367043479136
    assert str(length.unit) == 'ang'


@pytest.vaspxml_available  # type: ignore
def test_nearest_structure_reference(vasp_path_vol, vasp_path):
    structure = load_known_material('argyrodite')

    assert vasp_path_vol.dims == vasp_path.dims

    nearest = vasp_path.path_over_structure(structure)

    assert all(s.label == '48h' for s in nearest)

    assert_allclose(nearest[0].coords, [3.145908, 8.107908, 5.200176])
    assert_allclose(nearest[20].coords, [1.816092, 6.778092, 5.200176])
    assert_allclose(nearest[-1].coords, [3.145908, 8.107908, 5.200176])


@pytest.vaspxml_available  # type: ignore
def test_optimal_n_paths(vasp_F_graph):
    paths = optimal_n_paths(
        F_graph=vasp_F_graph,
        start=(10, 4, 13),
        stop=(21, 3, 10),
        n_paths=3,
        min_diff=0.1,
    )

    assert len(paths) == 3
    assert isclose(sum(paths[-1].energy), 5.351758190646607)
    assert sum(paths[-1].energy) > sum(paths[-2].energy)
