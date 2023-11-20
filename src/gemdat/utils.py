"""This module connects generally useful utilties."""
from __future__ import annotations

import warnings

import numpy as np
from pymatgen.core import Lattice, Structure
from scipy import signal


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
    """Backward fill values equal to `val` with upcoming values.

    See ffill for options.
    """
    if axis == 0:
        return bfill(arr.T).T

    if arr.ndim > 2:
        raise ValueError

    return np.fliplr(ffill(np.fliplr(arr), fill_val=fill_val))


def meanfreq(x: np.ndarray, fs: float = 1.0) -> np.ndarray:
    """Estimates the mean frequency in terms of the sample rate, fs.

    Vectorized version of https://stackoverflow.com/a/56487241

    Parameters
    ----------
    x : np.ndarray[i, j]
        Time series of measurement values. The mean frequency is computed
        along the last axis (-1).
    fs : float, optional
        Sampling frequency of the `x` time series. Defaults to 1.0.

    Returns
    -------
    mnfreq : np.ndarray
        Array of mean frequencies.
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)

    assert x.ndim == 2

    f, Pxx_den = signal.periodogram(x, fs, axis=-1)
    width = np.tile(f[1] - f[0], Pxx_den.shape)
    P = Pxx_den * width
    pwr = np.sum(P, axis=1).reshape(-1, 1)

    f = f.reshape(1, -1)

    mnfreq = np.dot(P, f.T) / pwr

    return mnfreq


def is_lattice_similar(a: Lattice | Structure,
                       b: Lattice | Structure,
                       length_tol: float = 0.5,
                       angle_tol: float = 1.0) -> bool:
    """Return True if lattices are similar within given tolerance.

    Parameters
    ----------
    a, b : pymatgen.core.lattice.Lattice | pymatgen.core.structure.Structure
        Input lattices or structures
    length_tol : float, optional
        Length tolerance in Angstrom
    angle_tol : float, optional
        Angle tolerance in degrees

    Returns
    -------
    bool
        Return True if lattices are similar
    """
    if isinstance(a, Structure):
        a = a.lattice
    if isinstance(b, Structure):
        b = b.lattice

    for a_length, b_length in zip(a.lengths, b.lengths):
        if abs(a_length - b_length) > length_tol:
            return False

    for a_angle, b_angle in zip(a.angles, b.angles):
        if abs(a_angle - b_angle) > angle_tol:
            return False

    return True


def warn_lattice_not_close(a: Lattice, b: Lattice):
    """Raises a userwarning if lattices are not close."""
    if not is_lattice_similar(a, b):
        warnings.warn(
            'Lattices are not similar.'
            f'a: {a.parameters}, b: {b.parameters}', UserWarning)
