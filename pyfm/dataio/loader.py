import os
import typing as t
from collections import namedtuple

from concurrent.futures import ThreadPoolExecutor
import h5py
import numpy as np
import pandas as pd

from pyfm.domain import LoadDictConfig, LoadH5Config

from pyfm.dataio.converter import data_to_frame
from pyfm.domain import WrappedDataPipe
from pyfm import utils

from functools import partial


dataFrameFn = t.Callable[[np.ndarray], pd.DataFrame]
loadFn = t.Callable[[str, t.Dict], pd.DataFrame]


def get_pickle_loader(filename: str, _: t.Dict, **kwargs):
    data = np.load(filename, allow_pickle=True)
    if isinstance(data, np.ndarray) and len(data.shape) == 0:
        data = data.item()

    # TODO: Debug this for when pickle file is just pure ndarray
    pickle_config = LoadDictConfig.create(**kwargs)

    return data_to_frame(data, pickle_config)


def get_csv_loader(filename: str, _: t.Dict[str, str], **kwargs):

    data = None
    return pd.read_csv(filename)


def get_hdf5_loader(filename: str, repl: t.Dict[str, str], **kwargs):
    """
    Loads data from an HDF5 file and returns it as a DataFrame.

    Args:
        filename (str): Path to the HDF5 file.
        repl (Dict[str, str]): A dictionary of string replacements to apply to the configuration.
        **kwargs: Additional keyword arguments passed to LoadH5Config.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.

    """

    data = None
    try:
        return pd.read_hdf(filename)
    except (ValueError, NotImplementedError):
        pass

    with h5py.File(filename) as file:
        h5_config = LoadH5Config.create(**kwargs).format_data_strings(repl)
        try:
            data = data_to_frame(file, h5_config)
        except ValueError:
            h5_config = h5_config.search_for_dataset_label(file)
            data = data_to_frame(file, h5_config)

    if data is not None:
        return data
    else:
        raise ValueError(f"File {filename} could not be loaded.")


def get_file_loader(file_path: str):
    ext = os.path.splitext(file_path)[1]

    match ext:
        case ".p" | ".npy":
            return get_pickle_loader
        case ".h5":
            return get_hdf5_loader
        case ".csv":
            return get_csv_loader
        case _:
            raise ValueError("File must have extension '.p' or '.h5'")


def load_files(
    filestem: str,
    replacements: t.Dict | None = None,
    regex: t.Dict | None = None,
    wildcard_fill: bool = False,
    aggregate: bool = False,
    skip_file_set: t.List[str] | None = None,
    **kwargs,
) -> WrappedDataPipe | pd.DataFrame:
    def file_loader_wrapper(file_loader, filename: str, repl: t.Dict) -> pd.DataFrame:
        utils.get_logger().debug(f"Loading file: {filename}")
        new_data: pd.DataFrame = file_loader(filename, repl)

        if len(repl) != 0:
            new_data[list(repl.keys())] = tuple(repl.values())

        return repl, new_data

    def file_factory():
        def get_filename(filename, reps):
            return filename, reps

        file_repls = utils.io.process_files(
            filestem, get_filename, replacements, regex, wildcard_fill
        )

        if skip_file_set:
            file_repls = [f for f in file_repls if f[0] not in skip_file_set]

        file_loader = partial(get_file_loader(filestem), **kwargs)
        flw = partial(file_loader_wrapper, file_loader)

        group_cols = []
        if len(file_repls) > 0:
            group_cols = file_repls[0][1].keys()

        GroupTuple = namedtuple("GroupTuple", group_cols)

        def temp(*args):
            fname, rep = args[0]
            return flw(fname, rep)

        with ThreadPoolExecutor(max_workers=1) as executor:
            results = executor.map(temp, file_repls)

        result_gen = ((GroupTuple(**g), r) for g, r in results)
        try:
            first = next(result_gen)
        except StopIteration:
            # No results
            yield (GroupTuple(), pd.DataFrame())
            return

        yield first
        yield from result_gen

    if aggregate:
        return WrappedDataPipe(file_factory).agg()
    else:
        return WrappedDataPipe(file_factory)
