"""This module connects generally useful utilties."""

from __future__ import annotations

import warnings
from importlib.resources import files
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure
from scipy import signal
from scipy.spatial import cKDTree

# shortcut to test data
VASPRUN = Path(__file__).parents[2] / 'tests' / 'data' / 'short_simulation' / 'vasprun.xml'
VASPCACHE_ORIENTATIONS = (
    Path(__file__).parents[2]
    / 'tests'
    / 'data'
    / 'short_simulation'
    / 'vasprun_rotations.cache'
)

DATA = files('gemdat') / 'data'


def nearest_structure_reference(structure: Structure) -> tuple[cKDTree, list[int]]:
    """Find distance and index of the nearest site of the structure for each
    voxel using a KD-tree.

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
        Structure of the material to use as reference for nearest site

    Returns
    -------
    kd_tree : scipy.spatial.cKDTree
        KD-tree of the structure
    periodic_ids : list[int]
        List of ids corresponding to the closest site of the structure
    """
    # In order to accomodate the periodicity, include the images of the structure sites
    periodic_structure = []
    periodic_ids: list[int] = []
    images = np.mgrid[-1:2, -1:2, -1:2].reshape(3, -1).T
    for dx, dy, dz in images:
        periodic_structure.extend(structure.frac_coords + np.array([dx, dy, dz]))

        # store the id of the site in the original structure
        periodic_ids.extend(range(len(structure.cart_coords)))

    # Create a KD-tree from the structure
    kd_tree = cKDTree(periodic_structure)

    return kd_tree, periodic_ids


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


def integer_remap(
    a: np.ndarray, key: np.ndarray, palette: np.ndarray | None = None
) -> np.ndarray:
    """Map integers in array `a` from `palette` -> `key`

    Parameters
    ----------
    a : np.ndarray
        Input array with values to be
    key : np.ndarray
        The key gives the new values that the palette will be mapped to
    palette : np.ndarray | None
        Input values, must be given in sorted order.
        If None, use sorted unique values in `a`

    Returns
    -------
    np.ndarray
    """
    if palette is None:
        palette = np.unique(a)

    index = np.digitize(a, palette, right=True)

    return key[index].reshape(a.shape)


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


def is_lattice_similar(
    a: Lattice | Structure,
    b: Lattice | Structure,
    length_tol: float = 0.5,
    angle_tol: float = 1.0,
) -> bool:
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
            f'Lattices are not similar.a: {a.parameters}, b: {b.parameters}',
            UserWarning,
        )


def _cart2sph(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    x : float
        x coordinate
    y : float
        y coordinate
    z : float
        z coordinate

    Returns
    -------
    az : float
        azimuthal angle
    el : float
        elevation angle
    r : float
        radius
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    el = np.arcsin(z / r)
    az = np.arctan2(y, x)
    return az, el, r


def cartesian_to_spherical(cart_coords: np.ndarray, *, degrees: bool = True) -> np.ndarray:
    """Trajectory from cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    cart_coords : np.ndarray
        Trajectory of the unit vectors in cartesian setting
    degrees : bool
        If true, return angles in degrees

    Returns
    -------
    spherical_coords : np.ndarray
        Trajectory of the unit vectors in spherical coordinates
    """
    x = cart_coords[:, :, 0]
    y = cart_coords[:, :, 1]
    z = cart_coords[:, :, 2]

    az, el, r = _cart2sph(x, y, z)

    if degrees:
        az = np.degrees(az)
        el = np.degrees(el)

    spherical_coords = np.stack((az, el, r), axis=-1)

    return spherical_coords


def fft_autocorrelation(coords: np.ndarray) -> np.ndarray:
    """Compute the autocorrelation of the given coordinates using FFT.

    Parameters
    ----------
    coords : np.ndarray
        The input signal in direct cartesian coordinates. It is expected
        to have shape (n_times, n_particles, n_coordinates)

    Returns
    -------
    autocorrelation: np.array
        The autocorrelation of the input signal, with shape (n_particles, n_times)
    """
    n_times, n_particles, n_coordinates = coords.shape

    autocorrelation = np.zeros((n_particles, n_times))
    normalization = np.arange(n_times, 0, -1)

    for c in range(n_coordinates):
        signal = coords[:, :, c]

        # Compute the FFT of the signal
        fft_signal = np.fft.rfft(signal, n=2 * n_times - 1, axis=0)
        # Compute the power spectral density in-place
        np.square(np.abs(fft_signal), out=fft_signal)
        # Compute the inverse FFT of the power spectral density
        autocorr_c = np.fft.irfft(fft_signal, axis=0)

        # Only keep the positive times
        autocorr_c = autocorr_c[:n_times, :]

        autocorrelation += autocorr_c.T / normalization

    # Normalize the autocorrelation such that it starts from 1
    # and make it independent of n_coordinates
    autocorrelation = autocorrelation / autocorrelation[:, 0, np.newaxis]

    return autocorrelation
