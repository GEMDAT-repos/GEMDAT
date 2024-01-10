from __future__ import annotations

from math import isclose

import pytest

from gemdat.io import load_known_material


@pytest.vaspxml_available  # type: ignore
def test_find_best_perc_path(vasp_full_path):

    assert isclose(vasp_full_path.cost, 36.39301483423)
    assert vasp_full_path.start_site == (30, 23, 14)


@pytest.vaspxml_available  # type: ignore
def test_nearest_structure_reference(vasp_full_vol, vasp_full_path):
    structure = load_known_material('argyrodite')
    vasp_full_vol.nearest_structure_reference(structure)

    vasp_full_path.fractional_path(vasp_full_vol)
    vasp_full_path.path_over_structure(structure, vasp_full_vol)

    assert isclose(vasp_full_path.frac_sites[55][0], 5.642595274217675)
    assert isclose(vasp_full_path.frac_sites[56][2], 2.0049868515758837)
    assert vasp_full_path.nearest_structure_label[0] == '48h'
    assert vasp_full_path.nearest_structure_label[10] == '48h'
    assert isclose(vasp_full_path.nearest_structure_coord[0][1],
                   9.685823999999998)
    assert isclose(vasp_full_path.nearest_structure_coord[28][2],
                   8.107907999999998)
    assert isclose(vasp_full_path.nearest_structure_coord[-1][-1],
                   1.816092000000001)
