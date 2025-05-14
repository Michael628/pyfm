import asyncio
import itertools
import logging
import os
import typing as t
from concurrent.futures import ThreadPoolExecutor
from functools import partial, singledispatch

import h5py
import numpy as np
import pandas as pd

import pyfm as ps
from pyfm import utils
from pyfm.processing import config, processor

dataFrameFn = t.Callable[[np.ndarray], pd.DataFrame]
loadFn = t.Callable[[str, t.Dict], pd.DataFrame]


def frame_to_frame(data, **y):
    assert isinstance(data, pd.DataFrame)
    return data


data_to_frame = singledispatch(frame_to_frame)


def ndarray_to_frame(
    array: np.ndarray, array_config: config.LoadArrayConfig
) -> pd.DataFrame:
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


def dict_to_frame(data: dict, dict_config: config.LoadDictConfig) -> pd.DataFrame:
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


def h5_to_frame(
    file: h5py.File,
    h5_config: config.LoadH5Config,
) -> pd.DataFrame:
    df = []
    for k, v in h5_config.datasets.items():
        dataset_label = (x for x in v if x in file)

        try:
            h5_dset = file[next(dataset_label)]
            assert isinstance(h5_dset, h5py.Dataset)
            data = h5_dset[:].view(np.complex128)
        except StopIteration:
            raise ValueError(f"dataset {k} not found in file.")

        frame = ndarray_to_frame(data, h5_config.array_config[k])
        frame[h5_config.name] = k
        df.append(frame)

    df = pd.concat(df)

    df.set_index(h5_config.name, append=True, inplace=True)

    return df


data_to_frame.register(h5_to_frame)


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


def get_pickle_loader(filename: str, _: t.Dict, **kwargs):
    data = np.load(filename, allow_pickle=True)
    if isinstance(data, np.ndarray) and len(data.shape) == 0:
        data = data.item()

    # TODO: Debug this for when pickle file is just pure ndarray
    pickle_config = config.LoadDictConfig.create(**kwargs)

    return data_to_frame(data, pickle_config)


def get_h5_loader(filename: str, repl: t.Dict[str, str], **kwargs):
    """
    Loads data from an HDF5 file and returns it as a DataFrame.

    Args:
        filename (str): Path to the HDF5 file.
        repl (Dict[str, str]): A dictionary of string replacements to apply to the configuration.
        **kwargs: Additional keyword arguments passed to LoadH5Config.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.

    """
    try:
        return pd.read_hdf(filename)
    except (ValueError, NotImplementedError):
        h5_config = config.LoadH5Config.create(**kwargs).format_data_strings(repl)

        file = h5py.File(filename)

        return h5_to_frame(file, h5_config)


def get_file_loader(filestem: str):
    if filestem.endswith(".p") or filestem.endswith(".npy"):
        return get_pickle_loader
    elif filestem.endswith(".h5"):
        return get_h5_loader
    else:
        raise ValueError("File must have extension '.p' or '.h5'")


async def load_files(
    filestem: str,
    replacements: t.Optional[t.Dict] = None,
    regex: t.Optional[t.Dict] = None,
    **kwargs,
) -> t.Dict[str, pd.DataFrame]:
    """
    Loads files and processes them using a provided processing function and optional replacements or regex operations.

    Parameters:
    filestem (str): The base name or stem of the files to be loaded and processed.
    file_loader (loadFn): A callable function responsible for file loading.
    replacements (t.Optional[t.Dict]): Optional dictionary containing key replacements to apply to `filestem`.
    regex (t.Optional[t.Dict]): Optional dictionary containing regex patterns as key replacements to `filestem`, matching
     all files found on disk according to the resulting pattern.

    Returns:
    pd.DataFrame: A pandas DataFrame result after processing the loaded files.
    """

    async def file_loader_wrapper(
        file_loader, filename: str, repl: t.Dict
    ) -> pd.DataFrame:
        logging.debug(f"Loading file: {filename}")
        new_data: pd.DataFrame = file_loader(filename, repl)

        if len(repl) != 0:
            new_data[list(repl.keys())] = tuple(repl.values())

        return new_data

    def get_filename(*args):
        return args

    file_repls = utils.process_files(filestem, get_filename, replacements, regex)

    file_loader = partial(get_file_loader(filestem), **kwargs)
    flw = partial(file_loader_wrapper, file_loader)

    async_tasks: t.List[asyncio.Task]
    async with asyncio.TaskGroup() as tg:
        async_tasks = [tg.create_task(flw(*fr)) for fr in file_repls]

    files = [f[0] for f in file_repls]
    return {file: t.result() for file, t in zip(files, async_tasks)}


def load(
    *args, **kwargs
) -> t.Dict[str, pd.DataFrame] | t.Coroutine[t.Any, t.Any, t.Dict[str, pd.DataFrame]]:
    loop = None
    try:
        # Try to get the running loop (will raise RuntimeError if not in a running loop)
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # If no running loop, use asyncio.run() as normal
        return asyncio.run(load_files(*args, **kwargs))

    assert loop, "We should have a loop or have already returned by now"

    if loop.is_running():
        return load_files(*args, **kwargs)
        # If inside a running loop (e.g., Jupyter), run the coroutine with `loop.create_task`
        # return loop.create_task(load_files(*args, **kwargs))
    else:
        # If not running, use `loop.run_until_complete`
        return loop.run_until_complete(load_files(*args, **kwargs))


def write_data(
    df: pd.DataFrame, filestem: str, write_fn: t.Callable[[pd.DataFrame, str], None]
) -> None:
    """
    Writes the data from a DataFrame to one or more files based on the given format and specified write function.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data to be written.
    filestem (str): A filename format string that may include placeholders for column values.
    write_fn (Callable[[pd.DataFrame, str], None]): A callable function responsible for writing each portion of the DataFrame to a file,
    taking the specific DataFrame segment and file path as arguments.

    Behavior:
    - If the `filestem` contains placeholders (keys enclosed in `{}`) that match column names in the DataFrame:
        - Validates that the DataFrame is non-empty and that all placeholder keys exist in the DataFrame columns.
        - Groups the DataFrame by the placeholder keys and writes each group to a separate file.
          The filename for each group is generated by replacing the placeholders in `filestem` with the corresponding group values.
        - The columns used for creating the groups are excluded from the output file, ensuring only the remaining columns are written.
    - If the `filestem` does not contain placeholders:
        - Writes the entire DataFrame to a single file using the specified `filestem` as the filename.
    - Creates the necessary directories for the output files if they do not already exist.
    """
    fs = os.path.expanduser(filestem)
    repl_keys = utils.format_keys(fs)
    if repl_keys:
        logging.debug(f"df columns: {df.columns}")
        logging.debug(f"df indices: {df.index.names}")
        assert len(df) != 0
        assert all([k in df.columns for k in repl_keys])

        for group, df_group in df.groupby(by=repl_keys):
            repl_vals = (group,) if isinstance(group, str) else group

            repl = dict(zip(repl_keys, repl_vals))

            filename = fs.format(**repl)
            logging.info(f"Writing file: {filename}")

            if directory := os.path.dirname(filename):
                os.makedirs(directory, exist_ok=True)

            out_cols = [c for c in df_group.columns if c not in repl_keys]

            write_fn(df_group[out_cols], filename)

    else:
        filename = fs
        logging.info(f"Writing file: {filename}")
        if directory := os.path.dirname(filename):
            os.makedirs(directory, exist_ok=True)

        write_fn(df, fs)


def write_dict(df: pd.DataFrame, filestem: str, dict_depth: int) -> None:
    """
    Writes a pandas DataFrame to a dictionary file format.

    Parameters:
    df: pd.DataFrame
        The input pandas DataFrame to be written.
    filestem: str
        The base name for the output file.
    dict_depth: int
        The depth of the dictionary structure to be created.

    See frame_to_dict for details on conversion from DataFrame to dictionary.
    """

    def writeconvert(data, fname):
        np.save(fname, frame_to_dict(data, dict_depth))

    write_data(df, filestem, write_fn=writeconvert)


def write_frame(df: pd.DataFrame, filestem: str) -> None:
    """
    Writes a DataFrame to a file with a specified filestem using a custom write function.

    Parameters:
    df : pd.DataFrame
        The DataFrame to be written to a file.
    filestem : str
        The base name for the output file.

    See write_data for details.
    """
    write_data(
        df,
        filestem,
        write_fn=lambda data, fname: data.to_hdf(fname, key="corr", mode="w"),
    )
