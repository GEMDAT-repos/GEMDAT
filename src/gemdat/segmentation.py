import numpy as np
from skimage.morphology._util import _offsets_to_raveled_neighbors
from skimage.segmentation import _watershed, _watershed_cy


def watershed_pbc(
    image,
    markers=None,
    connectivity=1,
    offset=None,
    mask=None,
    compactness=0,
    watershed_line=False,
):
    """Modified immplementation of [skimage.segmentation.watershed][] that adds
    periodic boundary conditions.

    See [watershed()][skimage.segmentation.watershed][] for details.
    """

    image, markers, mask = _watershed._validate_inputs(image, markers, mask,
                                                       connectivity)
    connectivity, offset = _watershed._validate_connectivity(
        image.ndim, connectivity, offset)

    mask = mask.ravel()
    output = markers.copy()

    flat_neighborhood = _offsets_to_raveled_neighbors(image.shape,
                                                      connectivity,
                                                      center=offset)
    marker_locations = np.flatnonzero(output)
    image_strides = np.array(image.strides, dtype=np.intp) // image.itemsize

    _watershed_cy.watershed_raveled(
        image.ravel(),
        marker_locations,
        flat_neighborhood,
        mask,
        image_strides,
        compactness,
        output.ravel(),
        watershed_line,
    )

    return output
