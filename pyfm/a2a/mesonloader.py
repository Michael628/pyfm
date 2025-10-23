"""
Meson field loading utilities for A2A contractions.

This module provides pure functions for efficiently loading meson field data
from HDF5 files for use in lattice QCD A2A contractions. The module supports
mass shifting, time slicing, and memory optimization through a singleton cache.

Key Features:
- Iterative loading of meson fields from multiple HDF5 files
- Support for mass shifting using eigenvalue files
- Time slice management for parallel processing
- Memory-efficient singleton caching
- Support for both CPU (numpy) and GPU (cupy) backends

Functions:
    iter_meson_fields: Main iterator for loading meson field data
    load_meson: Load a single meson field from HDF5
    meson_mass_alter: Apply mass shifting to a meson field
    get_index_range: Determine index range from contraction element
    get_meson_cache: Access the global cache
    clear_meson_cache: Clear the global cache
"""

import logging
import typing as t
from time import perf_counter

import h5py

try:
    import cupy as xp
except ImportError:
    import numpy as xp

from pyfm.domain import DiagramConfig, MesonLoaderConfig


_MESON_CACHE: dict[tuple[str, slice], tuple[slice, xp.ndarray]] = {}


def get_meson_cache() -> dict:
    """Get the global meson matrix cache."""
    return _MESON_CACHE


def clear_meson_cache() -> None:
    """Clear the global meson matrix cache."""
    global _MESON_CACHE
    _MESON_CACHE.clear()


def meson_mass_alter(mat: xp.ndarray, meson_config: MesonLoaderConfig) -> None:
    """
    Apply mass shifting to a meson field matrix using eigenvalue reweighting.

    This function modifies the input matrix in-place by applying a mass shift
    transformation. The transformation uses eigenvalue data to reweight the
    meson field from the original mass to a target mass.
    This is commonly used in lattice QCD to study mass dependence without
    regenerating expensive meson field data.

    The mass shifting is performed by:
    1. Loading eigenvalue data from the specified evalfile
    2. Computing scaling factors based on the mass shift
    3. Applying the scaling factors to the meson field matrix

    Parameters
    ----------
    mat : xp.ndarray
        Input meson field matrix to be modified in-place.
        Expected shape is (time, w_index, v_index) where w_index and v_index
        correspond to source and sink vector indices.
    meson_config : MesonLoaderConfig
        Configuration containing mass shift parameters.

    Raises
    ------
    KeyError
        If eigenvalue data is not found in the expected HDF5 dataset locations.
    FileNotFoundError
        If the evalfile cannot be opened.

    Notes
    -----
    - The matrix is modified in-place for memory efficiency
    - Eigenvalues are assumed to be stored as float64 in the HDF5 file
    - The method handles both standard and MILC mass conventions
    - Supports both '/evals' and '/EigenValueFile/evals' dataset paths
    - The scaling preserves the complex structure of the meson field

    Mathematical Details
    --------------------
    The mass shift scaling factor is computed as:
    scaling = (m_old + i*λ) / (m_new + i*λ)
    where λ are the eigenvalues and m_old, m_new are the old and new masses.
    For MILC convention, masses are multiplied by 2.
    """
    evalfile = meson_config.evalfile.filename
    oldmass = meson_config.mass[meson_config.mass_shift.original]
    newmass = meson_config.mass[meson_config.mass_shift.updated]
    milc_mass = meson_config.mass_shift.milc_mass

    with h5py.File(evalfile, "r") as f:
        try:
            evals = xp.array(f["/evals"][()].view(xp.float64), dtype=xp.float64)
        except KeyError:
            evals = xp.array(
                f["/EigenValueFile/evals"][()].view(xp.float64), dtype=xp.float64
            )

    evals = xp.sqrt(evals)

    mult_factor = 2.0 if milc_mass else 1.0
    eval_scaling = xp.zeros((len(evals), 2), dtype=xp.complex128)
    eval_scaling[:, 0] = xp.divide(
        mult_factor * oldmass + 1.0j * evals,
        mult_factor * newmass + 1.0j * evals,
    )
    eval_scaling[:, 1] = xp.conjugate(eval_scaling[:, 0])
    eval_scaling = eval_scaling.reshape((-1,))

    mat[:] = xp.multiply(mat, eval_scaling[xp.newaxis, xp.newaxis, :])


def load_meson(
    file: str,
    meson_config: MesonLoaderConfig,
    vmax_index: int | None,
    wmax_index: int | None,
    time: slice = slice(None),
) -> xp.ndarray:
    """
    Load meson field data from an HDF5 file with optional mass shifting.

    This function reads a 3-dimensional meson field array from an HDF5 file,
    applying time slicing and optional mass shifting. The data is automatically
    promoted from single to double precision for numerical accuracy in
    subsequent calculations.

    Parameters
    ----------
    file : str
        Path to the HDF5 file containing meson field data.
        The file should contain a group with an 'a2aMatrix' dataset.
    meson_config : MesonLoaderConfig
        Configuration containing mass shift and file parameters.
    vmax_index : int | None
        Maximum v-index to load (None = load all).
    wmax_index : int | None
        Maximum w-index to load (None = load all).
    time : slice, optional, default=slice(None)
        Time slice object specifying which time slices to load.
        Default loads all available time slices.

    Returns
    -------
    xp.ndarray
        3-dimensional complex array with shape (time, w_index, v_index).
        Data type is complex128 (double precision complex).
        The array contains the meson field data for the specified time range.

    Raises
    ------
    FileNotFoundError
        If the specified HDF5 file cannot be found or opened.
    KeyError
        If the expected 'a2aMatrix' dataset is not found in the file.
    ValueError
        If the data cannot be properly converted or shaped.

    Notes
    -----
    - Input data is assumed to be single precision complex (complex64)
    - Output is promoted to double precision complex (complex128)
    - Time slicing is applied first, followed by w/v index slicing
    - If mass_shift is configured, mass shifting is applied after loading
    - Loading time is logged at debug level for performance monitoring
    - The method handles the HDF5 file structure automatically

    Performance Notes
    -----------------
    - Loading time is measured and logged for performance analysis
    - Memory usage scales with the size of the requested time slice
    - For GPU backends, data is loaded directly to GPU memory when possible

    File Format Requirements
    ------------------------
    The HDF5 file should have the following structure:
    - Root level contains one or more groups
    - Each group contains an 'a2aMatrix' dataset
    - The dataset should have shape (time, w_max, v_max)
    - Data should be stored as complex64 (single precision complex)
    """
    t1 = perf_counter()

    with h5py.File(file, "r") as f:
        a_group_key = list(f.keys())[0]

        temp = f[a_group_key]["a2aMatrix"]
        temp = xp.array(
            temp[time, slice(wmax_index), slice(vmax_index)].view(xp.complex64),
            dtype=xp.complex128,
        )

    t2 = perf_counter()
    logging.debug(f"Loaded array {temp.shape} in {t2 - t1} sec")

    shift_mass = meson_config.mass_shift.updated is not None
    if shift_mass:
        fact = "2*" if meson_config.mass_shift.milc_mass else ""
        oldmass = meson_config.mass[meson_config.mass_shift.original]
        newmass = meson_config.mass[meson_config.mass_shift.updated]
        logging.info(f"Shifting mass from {fact}{oldmass:f} to {fact}{newmass:f}")
        meson_mass_alter(temp, meson_config)

    return temp


def get_index_range(index_str: str, diagram_config: DiagramConfig) -> int | None:
    """
    Determine the max index value based on contraction element.

    Parameters
    ----------
    index_str : str
        Contraction index string ('e' for eigenmode, or numeric for stochastic).
    diagram_config : DiagramConfig
        Diagram configuration containing eig_range and stoch_range.

    Returns
    -------
    int | None
        Maximum index value, or None for no limit.
    """
    if index_str == "e":
        if diagram_config.eig_range is not None:
            return diagram_config.eig_range.max
    else:
        if diagram_config.stoch_range is not None:
            return diagram_config.stoch_range.max
    return None


def iter_meson_fields(
    diagram_config: DiagramConfig,
    mesonfiles: tuple[str, ...],
    times: tuple,
    contraction: tuple[str, ...],
) -> t.Iterator[tuple[tuple[slice, xp.ndarray], ...]]:
    """
    Iterate over meson field data for all files across time slices.

    This function implements the iterator protocol, returning meson field data
    for all configured files at the current time step. It includes intelligent
    caching to avoid redundant file I/O operations when the same data is
    needed multiple times.

    The function implements several optimization strategies:
    1. Reuses data from previous iterations when time slices match
    2. Shares data between files when they contain identical time slices
    3. Only loads new data when necessary

    Parameters
    ----------
    diagram_config : DiagramConfig
        Diagram configuration containing meson loader configs and ranges.
    mesonfiles : tuple[str, ...]
        Tuple of HDF5 file paths to load meson data from.
    times : tuple
        Tuple of time slice lists, one per meson file.
    contraction : tuple[str, ...]
        Contraction tuple specifying indices (e.g., ('e', 'e', '0', '1')).

    Yields
    ------
    tuple[tuple[slice, xp.ndarray], ...]
        Tuple containing (time_slice, meson_data) pairs for each configured file.
        The length of the tuple equals the number of mesonfiles.
        Each meson_data array has shape (time, w_index, v_index).

    Notes
    -----
    - Data is cached in the global singleton cache to avoid redundant loading
    - Memory optimization through intelligent data sharing
    - Logging at debug level tracks loading and caching operations
    - The function handles both file-based and memory-based data reuse

    Caching Strategy
    ----------------
    The function uses a multi-level caching approach:
    1. Check if data from previous iteration can be reused (same time slice)
    2. Check if data from another file in current iteration can be shared
    3. Load new data from file only if neither cache hit occurs

    Performance Considerations
    --------------------------
    - File I/O is minimized through aggressive caching
    - Memory usage is optimized by sharing references when possible
    - Debug logging helps identify performance bottlenecks
    - Large arrays are only loaded when absolutely necessary

    Examples
    --------
    >>> for meson_data in iter_meson_fields(diagram_config, files, times, contraction):
    ...     (t1, data1), (t2, data2) = meson_data
    ...     # Process data1 and data2 for time slices t1 and t2
    """
    cache = get_meson_cache()

    for iter_idx in range(len(times[0])):
        current_times = [times[i][iter_idx] for i in range(len(mesonfiles))]
        result = []

        for i, (time, file) in enumerate(zip(current_times, mesonfiles)):
            cache_key = (file, time)

            if cache_key in cache:
                logging.debug(f"Using cached {time} from {file}")
                result.append(cache[cache_key])
                continue

            found = False
            for j in range(i):
                if current_times[j] == time and mesonfiles[j] == file:
                    cache_key_j = (mesonfiles[j], current_times[j])
                    if cache_key_j in cache:
                        logging.debug(f"Found {time} at index {j}")
                        result.append(cache[cache_key_j])
                        found = True
                        break

            if not found:
                meson_idx = i % len(diagram_config.mesons)
                meson_config = diagram_config.mesons[meson_idx]

                w_idx_str = contraction[i * 2]
                v_idx_str = contraction[i * 2 + 1]

                wmax_index = get_index_range(w_idx_str, diagram_config)
                vmax_index = get_index_range(v_idx_str, diagram_config)

                logging.debug(
                    f"Loading {time} from {file} (wmax={wmax_index}, vmax={vmax_index})"
                )
                data = load_meson(file, meson_config, vmax_index, wmax_index, time)
                cache[cache_key] = (time, data)
                result.append((time, data))

        yield tuple(result)
