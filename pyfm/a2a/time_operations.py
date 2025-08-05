"""Time-related operations for A2A contractions."""
import typing as t

try:
    import cupy as xp
except ImportError:
    import numpy as xp


def convert_to_numpy(corr: xp.ndarray):
    """Converts a cupy array to a numpy array"""
    if hasattr(xp, "asnumpy"):
        return xp.asnumpy(corr)
    else:
        return corr


def time_average(cij: xp.ndarray, open_indices: t.Tuple = (0, -1)) -> xp.ndarray:
    """Takes an array with dim >= 2 and returns an array of (dim-1) where the
    i-th element in the last axis of the array is the sum of all input elements
    separated by i in the axes specified by `open_indices`

    Parameters
    ----------
    cij : ndarray
        A dim >= 2 array

    open_indices : tuple, optional
        axis indices to average over. Defaults to first and last index.
        Assumes both indices have same length

    Returns
    -------
    ndarray
        Array matching `cij` after averaging. Last dimension of output is
        average over separations in `open_indices` axes
        using periodic boundary.
    """

    cij = xp.asarray(cij)  # Remain cp/np agnostic for utility functions

    nt = cij.shape[open_indices[0]]
    dim = len(cij.shape)

    ones = xp.ones(cij.shape)
    t_range = xp.array(range(nt))

    t_start = [None] * dim
    t_start[open_indices[0]] = slice(None)
    t_start = tuple(t_start)

    t_end = [None] * dim
    t_end[open_indices[1]] = slice(None)
    t_end = tuple(t_end)

    t_mask = xp.mod(t_range[t_end] * ones - t_range[t_start] * ones, xp.array([nt]))

    time_removed_indices = tuple(
        slice(None) if t_start[i] == t_end[i] else 0 for i in range(dim)
    )

    corr = xp.zeros(cij[time_removed_indices].shape + (nt,), dtype=xp.complex128)

    corr[:] = xp.array([cij[t_mask == t].sum() for t in range(nt)])

    return convert_to_numpy(corr / nt)