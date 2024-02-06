from __future__ import annotations

import numpy as np

from gemdat.shape import ShapeData


def test_shape(vasp_shape_data):
    assert len(vasp_shape_data) == 1

    for shape in vasp_shape_data:
        assert isinstance(shape, ShapeData)
        assert shape.name == '48h'
        assert shape.coords.shape == (19284, 3)

        # coords may not exceed threshold
        assert np.all(shape.coords > -1)
        assert np.all(shape.coords < 1)
