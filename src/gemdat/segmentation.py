from __future__ import annotations

from typing import Sequence

import numpy as np
from skimage.segmentation import watershed as skimage_watershed


def unpad(array: np.ndarray, pad_width: int | Sequence[int]) -> np.ndarray:
    """Opposite of `np.pad`."""
    slices = []
    if isinstance(pad_width, int):
        pad_width = [pad_width for _ in range(array.ndim)]

    for start in pad_width:
        end = -start
        slices.append(slice(start, end))

    return array[tuple(slices)]


def watershed_pbc(
    image: np.ndarray,
    markers: np.ndarray,
    *,
    mask: np.ndarray,
    pad: int = 10,
    **kwargs,
):
    """Modified version of [watershed()][skimage.segmentation.watershed] that
    wraps around the edges to account for periodic boundary conditions.

    See [watershed()][skimage.segmentation.watershed] for details on parameters.

    Parameters
    ----------
    pad : int
        Number of values to be padded around the edges of the image/markers/mask arrays.

    Returns
    -------
    out : np.ndarray
        Segmented array
    """
    image = np.pad(image, pad_width=pad, mode='wrap')
    mask = np.pad(mask, pad_width=pad, mode='wrap')
    markers = np.pad(markers, pad_width=pad, mode='wrap')

    out = skimage_watershed(image=image, markers=markers, mask=mask)

    return unpad(out, pad_width=pad)
