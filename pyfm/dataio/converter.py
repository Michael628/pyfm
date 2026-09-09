import dataclasses
import itertools
import typing as t
from functools import singledispatch

import h5py
import numpy as np
import pandas as pd

from pyfm.domain.io import LoadArrayConfig, LoadDictConfig, LoadH5Config


def frame_to_frame(data, **_):
    assert isinstance(data, pd.DataFrame)
    return data


data_to_frame = singledispatch(frame_to_frame)


def ndarray_to_frame(array: np.ndarray, array_config: LoadArrayConfig) -> pd.DataFrame:
    """
    Converts a multidimensional numpy array into a pandas DataFrame with a MultiIndex.

    Parameters:
    array : np.ndarray
        The input numpy array to be transformed into a DataFrame.
    array_params : config.LoadArrayConfig
        The configuration object that provides the order of dimensions (order),
        and labels for indexing each dimension.

    Returns:
    pd.DataFrame
        A DataFrame where the rows are indexed by a MultiIndex created based on
        the combinations of labels defined in array_params, and the single column
        'corr' holds the values from the input array.

    Raises:
    AssertionError
        If the number of label sets in array_params does not match the number
        of dimensions defined in array_params.order.

    Behavior:
    - If the array_params.order contains only 't', the function generates
      sequential labels for the 't' dimension.
    - Constructs a MultiIndex using the Cartesian product of the label sets
      specified in array_params.labels, adhering to the order in array_params.order.
    - Flattens the input array and pairs its values to the MultiIndex, creating
      a single-column DataFrame.
    """

    if len(array_config.order) == 1 and array_config.order[0] == "t":
        array_config.labels["t"] = list(range(np.prod(array.shape)))

    assert len(array_config.labels) == len(array_config.order)

    indices = [array_config.labels[k] for k in array_config.order]

    index: pd.MultiIndex = pd.MultiIndex.from_tuples(
        itertools.product(*indices), names=array_config.order
    )

    return pd.Series(array.reshape((-1,)), index=index, name="corr").to_frame()


data_to_frame.register(np.ndarray)(ndarray_to_frame)


def dict_to_frame(data: dict, dict_config: LoadDictConfig) -> pd.DataFrame:
    def entry_gen(
        nested: t.Dict, _index: t.Tuple = ()
    ) -> t.Generator[t.Tuple[t.Tuple, np.ndarray], None, None]:
        """Recursive Depth first search of nested dictionaries building
        list of indices from dictionary keys.


        Parameters
        ----------
            nested: dict
                The current sub-dictionary from traversing path `_index`
            _index: tuple(str)
                The sequence of keys traversed thus far in the original
                dictionary

        Yields
        ------
        (path, data)
            path: tuple(str)
                The sequence of keys traversed to get to `data` in
                the nested dictionary.
            data: ndarray
                The data that was found in `nested` by traversing indices
                in `path`.
        """

        if isinstance(next(iter(nested.values())), np.ndarray):
            assert all((isinstance(n, np.ndarray) for n in nested.values()))

            for key, val in nested.items():
                yield (_index + (key,), val)
        else:
            for key in nested.keys():
                yield from entry_gen(nested[key], _index + (key,))

    indices, concat_data = zip(*((index, array) for index, array in entry_gen(data)))
    concat_data = [data_to_frame(x, dict_config.array_config) for x in concat_data]

    for index, frame in zip(indices, concat_data):
        frame[dict_config.labels] = list(index)

    df = pd.concat(concat_data)

    df.set_index(dict_config.labels, append=True, inplace=True)

    return df


data_to_frame.register(dict_to_frame)


@dataclasses.dataclass(frozen=True)
class Hdf5EntryTemplate:
    """Per-dataset-key portion of the HDF5 index template.

    Holds everything that is invariant across files for one dataset key:
    the config key, the resolved h5 dataset label, and the pre-built index.
    The index is read-only and shared by reference across files (and, later,
    across threads) — it is never mutated by `_fill_hdf5_frame`.
    """

    key: str
    dataset_label: str
    index: pd.MultiIndex


@dataclasses.dataclass(frozen=True)
class Hdf5FrameTemplate:
    """Build-once template for an HDF5 run-key.

    Built once per run-key from an invariant `LoadH5Config` (the only HDF5
    producer emits brace-free literals, so `format_data_strings` is a no-op and
    the resulting `LoadH5Config` is identical for every file in the batch).
    """

    name: str
    entries: t.Tuple[Hdf5EntryTemplate, ...]


def build_hdf5_template(h5_config: LoadH5Config) -> Hdf5FrameTemplate:
    """Build the per-dataset-key MultiIndex template ONCE per run-key.

    Replaces the per-file `pd.MultiIndex.from_tuples` construction that
    `ndarray_to_frame` performed via the old `hdf5_to_frame`. The `order`/`labels`
    come from `LoadArrayConfig` and are invariant across files; the shape-derived
    `["dt"]` branch from `ndarray_to_frame` is intentionally absent here because
    the HDF5 producer emits `order=["t"]` (that branch is unreachable on this path).
    """
    entries = []
    for k, v in h5_config.datasets.items():
        assert (
            len(v) == 1
        ), "Only supporting single h5 entry per h5 dataset configuration key"
        dataset_label = v[0]
        array_config = h5_config.array_config[k]
        indices = [array_config.labels[dim] for dim in array_config.order]
        index: pd.MultiIndex = pd.MultiIndex.from_tuples(
            itertools.product(*indices), names=array_config.order
        )
        entries.append(
            Hdf5EntryTemplate(key=k, dataset_label=dataset_label, index=index)
        )
    return Hdf5FrameTemplate(name=h5_config.name, entries=tuple(entries))


def _read_hdf5_array(file: h5py.File, dataset_label: str) -> np.ndarray:
    """Read a complex128 ndarray from an h5 dataset or, failing that, a dataset attr."""
    if dataset_label in file:
        return file[dataset_label][:].view(np.complex128)
    base, attr_label = dataset_label.rsplit("/", 1)
    if base in file and attr_label in file[base].attrs:
        return file[base].attrs[attr_label][:].view(np.complex128)
    raise ValueError(f"dataset {dataset_label!r} not found in file.")


def _fill_hdf5_frame(file: h5py.File, template: Hdf5FrameTemplate) -> pd.DataFrame:
    """Per-file fill: read arrays and assemble frames using the pre-built index.

    This is the per-file hot path. It performs NO `MultiIndex` construction —
    the cached `entry.index` is reused by reference. The h5py read releases the
    GIL; the frame assembly is O(1) index reuse + `pd.concat` of the per-key
    frames (mirrors the old inner concat).
    """
    frames = []
    for entry in template.entries:
        data = _read_hdf5_array(file, entry.dataset_label)
        frame = pd.Series(
            data.reshape((-1,)), index=entry.index, name="corr"
        ).to_frame()
        frame[template.name] = entry.key
        frames.append(frame)
    df = pd.concat(frames)
    df.set_index(template.name, append=True, inplace=True)
    return df


def hdf5_to_frame(
    file: h5py.File,
    h5_config: LoadH5Config,
) -> pd.DataFrame:
    """Convert an h5py.File to a DataFrame (backward-compat wrapper).

    Builds the index template from `h5_config` and fills it from `file`. New call
    sites should build the template once via `build_hdf5_template` and reuse it
    across files via `_fill_hdf5_frame`.
    """
    return _fill_hdf5_frame(file, build_hdf5_template(h5_config))


data_to_frame.register(hdf5_to_frame)


def frame_to_dict(df: pd.DataFrame, dict_depth: int) -> t.Union[t.Dict, np.ndarray]:
    """
    Converts a pandas DataFrame into a dictionary or a numpy array depending on the specified depth.

    Parameters:
    df : pandas.DataFrame
        Input DataFrame to be converted.
    dict_depth : int
        Depth of the dictionary to create. If 0, a numpy array is returned instead of a dictionary.

    Returns:
    Union[Dict, numpy.ndarray]
        If `dict_depth` is 0, returns a multidimensional numpy array reshaped based on the levels of the index.
        Otherwise, returns a dictionary keyed by the concatenated indices up to the `dict_depth` level, and values
        are reshaped numpy arrays based on the remaining index levels.

    Behavior:
        Ensures that the `dict_depth` is within the permissible range (0 to the number of index levels).
        Reshapes the values of the DataFrame (assumed column name 'corr') into a multidimensional numpy array
        based on the levels of the index exceeding the specified `dict_depth`.
        Concatenates multi-level index keys up to the specified depth into a string format using '.' as a separator
        when returning a dictionary.

    Important Notes:
        - The DataFrame is expected to have a multi-index and a column named 'corr'.
        - If dict_depth exceeds the number of index levels or is negative, the function raises an assertion error.
        - The DataFrame is sorted based on the index before processing to ensure consistency in the output.
    """
    num_indices = len(df.index.names)
    assert dict_depth >= 0
    assert dict_depth <= num_indices

    shape = [
        len(df.index.get_level_values(i).drop_duplicates())
        for i in range(dict_depth, num_indices)
    ]
    shape = tuple([-1] + shape) if dict_depth != 0 else tuple(shape)

    keys = [
        df.sort_index().index.get_level_values(i).drop_duplicates().to_list()
        for i in range(dict_depth)
    ]

    def join_str_fn(x):
        return ".".join(map(str, x))

    keys = list(map(join_str_fn, list(itertools.product(*keys))))

    array = df.sort_index()["corr"].to_numpy().reshape(shape)

    if dict_depth == 0:
        return array
    else:
        return {k: array[i] for k, i in zip(keys, range(len(array)))}
