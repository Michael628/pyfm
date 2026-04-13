import os
import typing as t

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

    return pd.read_csv(filename)


def get_parquet_loader(filename: str, _: t.Dict[str, str], **kwargs):
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise NotImplementedError(
            "Parquet support requires pyarrow. Install with: pip install pyfm[parquet]"
        )

    return pd.read_parquet(filename)


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
        except ValueError as e:
            utils.get_logger().debug(f"Error loading HDF5 file: {e}")
            raise
            # h5_config = h5_config.search_for_dataset_label(file)
            # data = data_to_frame(file, h5_config)

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
        case ".parquet":
            return get_parquet_loader
        case _:
            raise ValueError("File must have extension '.p', '.h5', '.csv', or '.parquet'")


def load_files(
    filestem: str | t.List[str],
    replacements: t.Dict | None = None,
    regex: t.Dict | None = None,
    wildcard_fill: bool = False,
    aggregate: bool = False,
    skip_file_set: t.List[str] | None = None,
    **kwargs,
) -> WrappedDataPipe | pd.DataFrame:
    def file_factory():
        file_repls = utils.io.process_files(
            filestem, lambda f, r: (f, r), replacements, regex, wildcard_fill
        )

        if skip_file_set:
            file_repls = [f for f in file_repls if f[0] not in skip_file_set]

        if not file_repls:
            file0 = filestem if isinstance(filestem, str) else filestem[0] + ", ..."
            raise ValueError(f"No files found for file search pattern: {file0}")

        file_loader = partial(get_file_loader(file_repls[0][0]), **kwargs)
        group_cols = list(file_repls[0][1].keys())
        GroupTuple = utils.create_group_tuple(*group_cols)

        for filename, repl in file_repls:
            utils.get_logger().debug(f"Loading file: {filename}")
            df = file_loader(filename, repl)
            if repl:
                df[list(repl.keys())] = tuple(repl.values())
            yield GroupTuple(**repl), df

    if aggregate:
        return WrappedDataPipe(file_factory).agg()
    else:
        return WrappedDataPipe(file_factory)
