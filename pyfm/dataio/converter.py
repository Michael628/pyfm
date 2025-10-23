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
    - If the array_params.order contains only 'dt', the function generates
      sequential labels for the 'dt' dimension.
    - Constructs a MultiIndex using the Cartesian product of the label sets
      specified in array_params.labels, adhering to the order in array_params.order.
    - Flattens the input array and pairs its values to the MultiIndex, creating
      a single-column DataFrame.
    """

    if len(array_config.order) == 1 and array_config.order[0] == "dt":
        array_config.labels["dt"] = list(range(np.prod(array.shape)))

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


def hdf5_to_frame(
    file: h5py.File,
    h5_config: LoadH5Config,
) -> pd.DataFrame:
    df = []
    for k, v in h5_config.datasets.items():
        assert len(v) == 1, (
            "Only supporting single h5 entry per h5 dataset configuration key"
        )
        dataset_label = v[0]
        if dataset_label in file:
            data = file[dataset_label][:].view(np.complex128)
        else:
            dataset_label, attr_label = dataset_label.rsplit("/", 1)
            if dataset_label in file and attr_label in file[dataset_label].attrs:
                data = file[dataset_label].attrs[attr_label][:].view(np.complex128)
            else:
                raise ValueError(f"dataset {k} not found in file.")

        frame = ndarray_to_frame(data, h5_config.array_config[k])
        frame[h5_config.name] = k
        df.append(frame)

    df = pd.concat(df)

    df.set_index(h5_config.name, append=True, inplace=True)

    return df


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
