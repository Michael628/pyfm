"""
Meson field loading utilities for A2A contractions.

This module provides the MesonLoader class, which is responsible for efficiently
loading meson field data from HDF5 files for use in lattice QCD A2A contractions.
The loader supports mass shifting, time slicing, and memory optimization through
intelligent caching and reuse of loaded data.

Key Features:
- Iterative loading of meson fields from multiple HDF5 files
- Support for mass shifting using eigenvalue files
- Time slice management for parallel processing
- Memory-efficient caching and data reuse
- Support for both CPU (numpy) and GPU (cupy) backends

Classes:
    MesonLoader: Main class for loading and iterating over meson field data

The MesonLoader is designed to work with the A2A contraction workflow where
multiple meson fields need to be loaded at different time slices for various
diagram calculations.
"""

import logging
import typing as t
from time import perf_counter

import h5py
from pydantic.dataclasses import dataclass

try:
    import cupy as xp
except ImportError:
    import numpy as xp


@dataclass
class MesonLoader:
    """
    Iterable object that loads meson fields for processing in A2A contractions.

    This class provides an efficient iterator interface for loading meson field data
    from HDF5 files. It supports advanced features like mass shifting, intelligent
    caching, and memory optimization for large-scale lattice QCD calculations.

    The loader works by iterating through time slices and loading the corresponding
    meson field data from multiple files. It implements caching to avoid redundant
    file I/O operations when the same data is needed multiple times.

    Parameters
    ----------
    mesonfiles : List[str]
        List of file paths to HDF5 files containing meson field data.
        Each file should contain an 'a2aMatrix' dataset with meson field data.
    times : Tuple
        Tuple of iterables containing time slice objects. Each element corresponds
        to the time slices to be loaded from the corresponding file in mesonfiles.
        The slices are used to load specific time ranges from the HDF5 files.
    shift_mass : bool, optional, default=False
        If True, performs mass shifting on the loaded meson fields using eigenvalue
        data. This is used to reweight meson fields from one quark mass to another.
        Requires evalfile, oldmass, and newmass to be specified.
    evalfile : str, optional, default=""
        Path to HDF5 file containing eigenvalue data for mass shifting.
        Required if shift_mass is True. The file should contain eigenvalues
        under '/evals' or '/EigenValueFile/evals' dataset.
    oldmass : float, optional, default=0
        Original quark mass used in the meson field calculation.
        Required if shift_mass is True.
    newmass : float, optional, default=0
        Target quark mass for mass shifting.
        Required if shift_mass is True.
    vmax_index : Optional[int], optional, default=None
        Maximum index for the 'v' (sink) dimension when loading data.
        If None, loads all indices available in meson file.
    wmax_index : Optional[int], optional, default=None
        Maximum index for the 'w' (source) dimension when loading data.
        If None, loads all indices available in meson file.
    milc_mass : bool, optional, default=True
        If True, applies MILC mass convention (factor of 2) in mass shifting.
        This accounts for the different mass normalization used in MILC.

    Attributes
    ----------
    mesonlist : List[Optional[Tuple]]
        Internal cache storing loaded meson data as (time_slice, data) tuples.
        Used to avoid redundant file loading operations.
    iter_count : int
        Internal counter tracking the current iteration position.

    Examples
    --------
    Basic usage for loading meson fields:

    >>> files = ['meson1.h5', 'meson2.h5']
    >>> time_slices = ([slice(0, 10), slice(10, 20)], [slice(0, 10), slice(10, 20)])
    >>> loader = MesonLoader(mesonfiles=files, times=time_slices)
    >>> for (t1, data1), (t2, data2) in loader:
    ...     # Process meson field data
    ...     pass

    Usage with mass shifting:

    >>> loader = MesonLoader(
    ...     mesonfiles=files,
    ...     times=time_slices,
    ...     shift_mass=True,
    ...     evalfile='eigenvals.h5',
    ...     oldmass=0.01,
    ...     newmass=0.02
    ... )

    Notes
    -----
    - The loader assumes HDF5 files contain complex64 data and promotes to complex128
    - Memory usage is optimized through intelligent caching and data reuse
    - Supports both CPU (numpy) and GPU (cupy) backends automatically
    - Time slices should be slice objects or compatible indexing objects
    """

    mesonfiles: t.List[str]
    times: t.Tuple
    shift_mass: bool = False
    evalfile: str = ""
    oldmass: float = 0
    newmass: float = 0
    vmax_index: t.Optional[int] = None
    wmax_index: t.Optional[int] = None
    milc_mass: bool = True

    def meson_mass_alter(self, mat: xp.ndarray) -> None:
        """
        Apply mass shifting to a meson field matrix using eigenvalue reweighting.

        This method modifies the input matrix in-place by applying a mass shift
        transformation. The transformation uses eigenvalue data to reweight the
        meson field from the original mass (oldmass) to a target mass (newmass).
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

        with h5py.File(self.evalfile, "r") as f:
            try:
                evals = xp.array(f["/evals"][()].view(xp.float64), dtype=xp.float64)
            except KeyError:
                evals = xp.array(
                    f["/EigenValueFile/evals"][()].view(xp.float64), dtype=xp.float64
                )

        evals = xp.sqrt(evals)

        mult_factor = 2.0 if self.milc_mass else 1.0
        eval_scaling = xp.zeros((len(evals), 2), dtype=xp.complex128)
        eval_scaling[:, 0] = xp.divide(
            mult_factor * self.oldmass + 1.0j * evals,
            mult_factor * self.newmass + 1.0j * evals,
        )
        eval_scaling[:, 1] = xp.conjugate(eval_scaling[:, 0])
        eval_scaling = eval_scaling.reshape((-1,))

        mat[:] = xp.multiply(mat, eval_scaling[xp.newaxis, xp.newaxis, :])

    def load_meson(self, file: str, time: slice = slice(None)) -> xp.ndarray:
        """
        Load meson field data from an HDF5 file with optional mass shifting.

        This method reads a 3-dimensional meson field array from an HDF5 file,
        applying time slicing and optional mass shifting. The data is automatically
        promoted from single to double precision for numerical accuracy in
        subsequent calculations.

        Parameters
        ----------
        file : str
            Path to the HDF5 file containing meson field data.
            The file should contain a group with an 'a2aMatrix' dataset.
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
        - If shift_mass is True, mass shifting is applied after loading
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
                temp[time, slice(self.wmax_index), slice(self.vmax_index)].view(
                    xp.complex64
                ),
                dtype=xp.complex128,
            )

        t2 = perf_counter()
        logging.debug(f"Loaded array {temp.shape} in {t2 - t1} sec")

        if self.shift_mass:
            fact = "2*" if self.milc_mass else ""
            logging.info(
                (f"Shifting mass from {fact}{self.oldmass:f} to {fact}{self.newmass:f}")
            )
            self.meson_mass_alter(temp)

        return temp

    def __iter__(self) -> "MesonLoader":
        """
        Initialize the iterator for loading meson field data.

        This method sets up the internal state for iteration, including
        initializing the cache for loaded meson data and resetting the
        iteration counter.

        Returns
        -------
        MesonLoader
            Returns self to enable the iterator protocol.

        Notes
        -----
        - Initializes mesonlist cache with None values
        - Resets iter_count to -1 to prepare for iteration
        - Can be called multiple times to restart iteration
        - Memory from previous iterations is cleared
        """
        self.mesonlist = [None for _ in range(len(self.mesonfiles))]
        self.iter_count = -1
        return self

    def __next__(self) -> t.Tuple[t.Tuple[slice, xp.ndarray], ...]:
        """
        Get the next set of meson field data for all files at the current iteration.

        This method implements the iterator protocol, returning meson field data
        for all configured files at the current time step. It includes intelligent
        caching to avoid redundant file I/O operations when the same data is
        needed multiple times.

        The method implements several optimization strategies:
        1. Reuses data from previous iterations when time slices match
        2. Shares data between files when they contain identical time slices
        3. Only loads new data when necessary

        Returns
        -------
        Tuple[Tuple[slice, xp.ndarray], ...]
            Tuple containing (time_slice, meson_data) pairs for each configured file.
            The length of the tuple equals the number of mesonfiles.
            Each meson_data array has shape (time, w_index, v_index).

        Raises
        ------
        StopIteration
            When all time slices have been processed, indicating the end of iteration.

        Notes
        -----
        - Data is cached in mesonlist to avoid redundant loading
        - Memory optimization through intelligent data sharing
        - Logging at debug level tracks loading and caching operations
        - The method handles both file-based and memory-based data reuse

        Caching Strategy
        ----------------
        The method uses a multi-level caching approach:
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
        >>> loader = MesonLoader(files, times)
        >>> for meson_data in loader:
        ...     (t1, data1), (t2, data2) = meson_data
        ...     # Process data1 and data2 for time slices t1 and t2
        """
        if self.iter_count < len(self.times[0]) - 1:
            self.iter_count += 1

            current_times = [
                self.times[i][self.iter_count] for i in range(len(self.mesonfiles))
            ]

            for i, (time, file) in enumerate(zip(current_times, self.mesonfiles)):
                try:
                    # Check if meson exists from last iter
                    if self.mesonlist[i] is not None:
                        if self.mesonlist[i][0] == time:
                            continue

                    # Check for matching time slice
                    matches = [
                        j
                        for j in range(len(current_times[:i]))
                        if current_times[j] == time
                    ]

                    # Check for matching file names
                    j = self.mesonfiles.index(file)

                    # Check that file matches desired time
                    if j not in matches:
                        raise ValueError

                    logging.debug(f"Found {time} at index {j}")
                    self.mesonlist[i] = (time, self.mesonlist[j][1])  # Copy reference

                except ValueError:
                    logging.debug(f"Loading {time} from {file}")
                    self.mesonlist[i] = (
                        time,
                        self.load_meson(file, time),
                    )  # Load new file

            return tuple(self.mesonlist)
        else:
            self.mesonlist = None
            raise StopIteration
