from __future__ import annotations

from math import isclose

import pytest

from gemdat.io import load_known_material
from gemdat.path import multiple_paths, optimal_path


@pytest.vaspxml_available  # type: ignore
def test_fractional_coordinates(vasp_path_vol, vasp_full_path):
    frac_sites = vasp_full_path.fractional_path(vasp_path_vol)

    assert isclose(frac_sites[-1][0], 0.39285714285714285)
    assert isclose(frac_sites[19][1], 0.6428571428571429)
    assert isclose(frac_sites[10][2], 0.42857142857142855)

    assert all(element < 1 for element in max(frac_sites))
    assert all(element > 0 for element in max(frac_sites))


TEST_DATA = (
    # start, stop, method, expected
    ((10, 4, 13), (21, 3, 10), 'dijkstra', 5.210130709736149),
    ((7, 9, 2), (20, 4, 2), 'bellman-ford', 5.5363545176821),
    ((18, 7, 12), (25, 3, 13), 'dijkstra-exp', 5.029753032964493),
    ((3, 3, 6), (0, 9, 4), 'minmax-energy', 2.708267913357463),
)


@pytest.vaspxml_available  # type: ignore
@pytest.mark.parametrize('start,stop,method,expected', TEST_DATA)
def test_optimal_path(vasp_F_graph, start, stop, method, expected):
    path = optimal_path(vasp_F_graph, start, stop, method)
    assert isclose(sum(path.energy), expected)


@pytest.vaspxml_available  # type: ignore
def test_find_best_perc_path(vasp_full_path):
    assert isclose(vasp_full_path.cost, 11.488013690080908)
    assert vasp_full_path.start_site == (11, 9, 6)


@pytest.vaspxml_available  # type: ignore
def test_nearest_structure_reference(vasp_full_vol, vasp_full_path):
    structure = load_known_material('argyrodite')

    nearest_structure_label, nearest_structure_coord = vasp_full_path.path_over_structure(
        structure, vasp_full_vol)

    assert nearest_structure_label[0] == '48h'
    assert nearest_structure_label[10] == '48h'

    assert isclose(nearest_structure_coord[0][0], 0.2381760000000003)
    assert isclose(nearest_structure_coord[0][1], 1.8160919999999998)
    assert isclose(nearest_structure_coord[20][2], 3.145908)
    assert isclose(nearest_structure_coord[-1][-1], 1.816092)


@pytest.vaspxml_available  # type: ignore
def test_multiple_paths(vasp_F_graph):
    paths = multiple_paths(vasp_F_graph,
                           start=(10, 4, 13),
                           end=(21, 3, 10),
                           Npaths=3,
                           min_diff=0.1)

    assert len(paths) == 3
    assert isclose(sum(paths[-1].energy), 5.351758190646607)
    assert sum(paths[-1].energy) > sum(paths[-2].energy)
