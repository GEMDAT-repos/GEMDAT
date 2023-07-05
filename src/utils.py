import numpy as np


def ffill(arr: np.ndarray, fill_val: int = -1, axis=-1) -> np.ndarray:
    """Forward fill values equal to `val` with most recent values.

    Parameters
    ----------
    arr : np.ndarray
        Input array with 2 dimensions
    fill_val : int, optional
        Value to fill
    axis : int, optional
        Axis along which to operate

    Returns
    -------
    out : np.ndarray
        Output array with all values
    """
    if axis == 0:
        return ffill(arr.T).T

    if arr.ndim > 2:
        raise ValueError

    idx = np.where(arr != fill_val, np.arange(arr.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    return arr[np.arange(idx.shape[0])[:, None], idx]


def bfill(arr: np.ndarray, fill_val: int = -1, axis=-1) -> np.ndarray:
    """Backward fill.

    See ffill for options.
    """
    if axis == 0:
        return bfill(arr.T).T

    if arr.ndim > 2:
        raise ValueError

    return np.fliplr(ffill(np.fliplr(arr), fill_val=fill_val))
