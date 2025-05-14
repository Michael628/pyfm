#! /usr/bin/env python3
import logging
import typing as t

import gvar as gv
import gvar.dataset as ds
import numpy as np
import pandas as pd
from pyfm.a2a import contract as a2a

ACTION_ORDER = [
    "build_high",
    "average",
    "sum",
    "time_average",
    "real",
    "permkey_split",
    "permkey_average",
    "permkey_normalize",
    "normalize",
    "index",
    "drop",
]


def col_mask_gen(df: pd.DataFrame, group_cols: list[str]) -> t.Generator[pd.Series]:
    """Yields a mask for each combination of columns in `group_cols`"""
    groupno = df.groupby(group_cols).ngroup()
    for i in range(groupno.nunique()):
        yield groupno == i


def stdjackknife(buff: gv.BufferDict) -> gv.BufferDict:
    """
    Compute the jackknife standard deviation for each element in a BufferDict.

    This function performs jackknife resampling to calculate the standard deviation
    for each element in the input BufferDict. Jackknife resampling involves systematically
    leaving out one sample at a time and computing the desired statistic on the remaining data.

    Parameters:
        buff (gv.BufferDict): Input BufferDict containing data arrays for which
                              the jackknife standard deviation is to be computed.

    Returns:
        gv.BufferDict: A BufferDict containing the computed standard deviations
                       for each key in the input BufferDict.
    """
    buff_std = gv.BufferDict()

    for k, v in buff.items():
        stdarray = np.empty_like(v)
        for i in range(len(v)):
            j_sample = np.delete(v, i, axis=0)
            stdarray[i, :] = np.std(j_sample, axis=0, ddof=1)
        buff_std[k] = stdarray

    return buff_std


def mask_apply(
    df: pd.DataFrame,
    func: t.Callable,
    data_col: str,
    ungrouped_cols: t.List,
    invert: bool = False,
) -> pd.DataFrame:
    all_cols = list(df.columns)

    if not invert:
        ungrouped = ungrouped_cols + [data_col]
        grouped = [x for x in all_cols if x not in ungrouped]
    else:
        grouped = ungrouped_cols
        ungrouped = [x for x in all_cols if x not in grouped] + [data_col]

    df_out = []

    for mask in col_mask_gen(df, grouped):
        df_out.append(func(df[mask]))

    return pd.concat(df_out)


def group_apply(
    df: pd.DataFrame,
    func: t.Callable,
    data_col: str,
    ungrouped_cols: t.List,
    invert: bool = False,
) -> pd.DataFrame:
    """Applies `func` to `data_col` in `df` grouped by `ungrouped_cols`.

    Parameters
    ----------
    df : pd.DataFrame

    func : t.Callable

    data_col : str

    ungrouped_cols : t.List

    invert : bool, optional


    Returns
    -------
    pd.DataFrame


    """
    all_cols = list(df.index.names) + list(df.columns)

    if not invert:
        ungrouped = ungrouped_cols + [data_col]
        grouped = [x for x in all_cols if x not in ungrouped]
    else:
        grouped = ungrouped_cols
        ungrouped = [x for x in all_cols if x not in grouped] + [data_col]

    df_out = df.reset_index().groupby(by=grouped, dropna=False)[ungrouped].apply(func)

    # while None in df_out.index.names:
    #     df_out = df_out.droplevel(df_out.index.names.index(None))

    return df_out


def buffer_avg(buff: gv.BufferDict) -> pd.DataFrame:
    """
    Computes the average of data stored in a BufferDict and returns it as a pandas DataFrame.

    Parameters:
        buff (gv.BufferDict): A dictionary-like object where keys are group identifiers
                              and values are numpy arrays of data.

    Returns:
        pd.DataFrame: A DataFrame containing the averaged data, with keys from the BufferDict
                      as column names.
    """

    avg_data = ds.avg_data(buff)
    df = pd.DataFrame(dict(avg_data))
    df.index.name = "t"
    return df


def buffer(
    df: pd.DataFrame,
    data_col: str,
    key_index: t.Union[str, t.List[str]],
    fold: bool = False,
) -> t.Tuple[gv.BufferDict, t.List[str]]:
    """
    Converts a pandas DataFrame into a grouped buffer dictionary.

    Parameters:
        df (pd.DataFrame): The input DataFrame, expected to have a multi-index with a time variable.
        data_col (str): The name of the column containing the data to be buffered.
        key_index (Union[str, List[str]]): The column(s) or index level(s) to group by.

    Returns:
        gv.BufferDict: A dictionary-like object where keys are group identifiers and values are
                       numpy arrays of reshaped data corresponding to each group.

    Raises:
        AssertionError: If the specified key_index is not found in the DataFrame's columns or index levels.
    """
    tvar = "t" if "t" in df.index.names else "dt"

    buff = gv.BufferDict()

    nt = df.index.get_level_values(tvar).nunique()

    labels_dt_last = sorted(df.index.names, key=lambda x: 0 if x == tvar else -1)

    if isinstance(key_index, str):
        key_indices = [key_index]
    else:
        key_indices = key_index

    if key_indices[0] in df.columns:
        assert all([x in df.columns for x in key_indices])
        group_param = {"by": key_indices}
    else:
        assert all([x in df.index.names for x in key_indices])
        group_param = {"level": key_indices}

    buff_fold = gv.BufferDict()

    for key, xs in df.groupby(**group_param):
        buff[key] = (
            xs.reorder_levels(labels_dt_last)[data_col].to_numpy().reshape((-1, nt))
        )
        if fold:
            size = buff[key].shape
            size = (size[0], size[1] // 2 + 1)

            buff_fold[key] = np.zeros(size, dtype=buff[key].dtype)

            buff_fold[key][:, 1 : nt // 2] = (
                buff[key][:, 1 : nt // 2] + buff[key][:, -1 : nt // 2 : -1]
            ) / 2.0
            buff_fold[key][:, 0] = buff[key][:, 0]
            buff_fold[key][:, nt // 2] = buff[key][:, nt // 2]

    return (buff if not fold else buff_fold, key_indices)


def melt_df_cols(
    df: pd.DataFrame,
    data_col: str,
    key_index: t.Union[str, t.List[str]],
    *args,
    **kwargs,
) -> pd.DataFrame:
    logging.info(data_col)
    logging.info(key_index)
    df_out = pd.melt(df, value_name=data_col, ignore_index=False)
    if isinstance(key_index, str) or len(key_index) == 1:
        var_names = ["variable"]
    else:
        var_names = [f"variable_{i}" for i in range(len(key_index))]

    df_out.rename(columns=dict(zip(var_names, key_index)), inplace=True)
    df_out[["mean", "error"]] = (
        df_out[data_col].apply(lambda x: (gv.mean(x), gv.sdev(x))).to_list()
    )
    return df_out


def signal_noise_nts(
    data: t.Union[gv.BufferDict, pd.DataFrame], stacked: bool, *args, **kwargs
) -> t.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    corr_col = "corr"
    if isinstance(data, pd.DataFrame):
        buff, col_names = buffer(data, *args, **kwargs)
    else:
        assert isinstance(data, gv.BufferDict)
        buff = data
        col_names = None

    signal = buffer_avg(buff)
    noise = buffer_avg(stdjackknife(buff))
    if col_names is not None:
        signal.columns.names = col_names
        noise.columns.names = col_names

    nts = noise.copy()
    import numpy as np

    for k in nts.columns:
        nts[k] = np.divide(nts[k], signal[k])

    # Merges data from columns into a single stack with [variable_i...] columns
    # giving the values of the buffer keys
    if stacked:
        signal, noise, nts = (
            melt_df_cols(df, *args, **kwargs) for df in (signal, noise, nts)
        )

    return signal, noise, nts


# def build_high(df: pd.DataFrame, data_col) -> pd.DataFrame:

#     high = df.xs('ama', level='dset').sort_index()[data_col] \
#         - df.xs('ranLL', level='dset').sort_index()[data_col]
#     high = high.to_frame(data_col)
#     high['dset'] = 'high'
#     high.set_index('dset', append=True, inplace=True)
#     high = high.reorder_levels(df.index.names)

#     return pd.concat([df, high])


def drop(df, data_col, *args):
    for key in args:
        assert isinstance(key, str)

        if key in df.index.names:
            df.reset_index(key, drop=True, inplace=True)
        elif key in df.columns:
            _ = df.pop(key)
        else:
            raise ValueError(f"Drop Failed - No index or column `{key}` found.")
    return df


def index(df, data_col: str, *args) -> pd.DataFrame:
    """
    Reorganizes the given DataFrame by resetting and setting specified indices.

    Parameters:
    - df (pd.DataFrame): The input DataFrame to be indexed.
    - data_col (str): A column name (currently unused in this function).
    - indices (List[str]): A list of column names to set as the new index.

    Returns:
    - pd.DataFrame: The DataFrame with the specified indices set and sorted.

    Notes:
    - If "series.cfg" is in the indices list but not in the DataFrame's index or columns,
      it is constructed by concatenating the "series" and "cfg" columns with a "." separator.
    - The function ensures that any existing indices are reset before setting new ones.
    - The DataFrame is sorted by the new index after setting it.

    Raises:
    - AssertionError: If any element in `indices` is not a string.
    - AssertionError: If "series" or "cfg" is not found in the DataFrame when "series.cfg" needs to be built.
    """
    indices = list(args)

    assert all([isinstance(i, str) for i in indices])

    if not indices:
        return df

    series_cfg = "series_cfg"
    # HACK: force series.cfg to become series_cfg
    if "series.cfg" in indices:
        i = indices.index("series.cfg")
        indices[i] = series_cfg
    df.rename_axis(index={"series.cfg": series_cfg}, inplace=True)
    df.rename({"series.cfg": series_cfg}, inplace=True)
    logging.debug(f"Current df index: {df.index.names}")
    logging.debug(f"Current df columns: {df.columns}")
    logging.debug(f"Setting index as {indices}")
    # End HACK

    build_seriescfg = series_cfg in indices
    build_seriescfg &= series_cfg not in df.index.names
    build_seriescfg &= series_cfg not in df.columns

    if build_seriescfg:
        series: pd.DataFrame
        cfg: pd.DataFrame
        for key in ["series", "cfg"]:
            if key in df.index.names:
                df.reset_index(key, inplace=True)
            else:
                assert key in df.columns

        series = df.pop("series")
        cfg = df.pop("cfg")

        df[series_cfg] = series + "." + cfg

        if series_cfg in df.index.names:
            df.reset_index(series_cfg, drop=True, inplace=True)

    df.reset_index(inplace=True)
    df.set_index(indices, inplace=True)
    df.sort_index(inplace=True)

    return df


def real(df, data_col, apply_real: bool = True):
    if apply_real:
        df[data_col] = df[data_col].apply(np.real)
    return df


def normalize(df, data_col, divisor):
    return df["corr"].apply(lambda x: x / float(divisor)).to_frame()


def sum(df: pd.DataFrame, data_col, *sum_indices) -> pd.DataFrame:
    """Sums `data_col` column in `df` over columns or indices specified in `avg_indices`"""
    return group_apply(df, lambda x: x[data_col].mean(), data_col, list(sum_indices))


def average(df: pd.DataFrame, data_col, *avg_indices) -> pd.DataFrame:
    """Averages `data_col` column in `df` over columns or indices specified in `avg_indices`,
    one at a time.
    """

    def average_repeated_indices(df):
        df_out = df.copy()
        df_out[data_col] = df.groupby(level=df.index.names)[data_col].mean()
        df_out = df_out.drop_duplicates()
        return df_out

    df_out = df
    for col in avg_indices:
        if col in df.index.names:
            df_out = df_out.reset_index(col, drop=True)
        else:
            df_out = df_out.drop(columns=col)

        df_out = mask_apply(df_out, average_repeated_indices, data_col, [])

    return df_out


def permkey_split_old(
    df: pd.DataFrame, data_col, permkey_col: str = "permkey"
) -> pd.DataFrame:
    df[permkey_col] = df[permkey_col].str.replace("e", "")
    df[permkey_col] = df[permkey_col].str.replace("v[0-9]+", ",", regex=True)
    df[permkey_col] = df[permkey_col].str.replace("w", "")
    df[permkey_col] = df[permkey_col].str.rstrip(",")
    df[permkey_col] = df[permkey_col].str.lstrip(",")
    key_len = df.iloc[0][permkey_col].count(",")
    assert all(df[permkey_col].str.count(",") == key_len)
    n_high = int(key_len + 1)

    df[[f"{permkey_col}{i}" for i in range(n_high)]] = df[permkey_col].str.split(
        ",", expand=True
    )
    df.drop(permkey_col, inplace=True, axis="columns")
    return df


def permkey_split(
    df: pd.DataFrame, data_col, permkey_col: str = "permkey"
) -> pd.DataFrame:
    if permkey_col in df.index.names:
        df.reset_index(permkey_col, inplace=True)

    if "_" not in df.iloc[0][permkey_col]:
        return permkey_split_old(df, data_col, permkey_col)

    df[permkey_col] = df[permkey_col].str.replace("(e_|_e)", "", regex=True)
    key_len = df.iloc[0][permkey_col].count("_")
    assert all(df[permkey_col].str.count("_") == key_len)
    n_high = int(key_len + 1) // 2

    df[[f"{permkey_col}{i}" for i in range(n_high)]] = df[permkey_col].str.split(
        "_", expand=True
    )[list(range(n_high))]
    df.drop(permkey_col, inplace=True, axis="columns")
    return df


def permkey_normalize(
    df: pd.DataFrame, data_col, permkey_col: str = "permkey"
) -> pd.DataFrame:
    df_out = df
    if f"{permkey_col}0" not in df_out.columns:
        df_out = permkey_split(df_out, data_col, permkey_col)

    perm_cols = [x for x in df_out.columns if permkey_col in x]

    n_high_modes = df_out[f"{permkey_col}{len(perm_cols) - 1}"].astype(int).max() + 1
    n_unique_comb = df_out[perm_cols].drop_duplicates()[f"{permkey_col}0"].count()
    n_index_modes = n_high_modes - (len(perm_cols) - 1)
    df_out[data_col] = df_out[data_col] * n_unique_comb / n_index_modes
    for p in perm_cols[:-1]:
        n_index_modes += 1
        df_out[data_col] = df_out[data_col] / (
            n_index_modes - df_out[p].astype(int) - 1
        )

    return df_out


def permkey_average(
    df: pd.DataFrame, data_col, permkey_col: str = "permkey"
) -> pd.DataFrame:
    df_out = permkey_split(df, data_col, permkey_col)

    perm_cols = [x for x in df_out.columns if permkey_col in x]

    return average(df_out, data_col, *perm_cols)


def time_average(df: pd.DataFrame, data_col: str, *avg_indices) -> pd.DataFrame:
    """Averages `data_col` column in `df` over columns or indices specified in `avg_indices`,
    one at a time.
    """
    assert len(avg_indices) == 2
    # tvar = "t" if "t" in df.index.names else "dt"
    # Old mysterious line of code above. Just in case this makes sense in some world, assert that it doesn't make sense.
    assert "t" not in df.index.names
    tvar = "t"

    def apply_func(x):
        nt = int(np.sqrt(len(x)))
        assert nt**2 == len(x)
        corr = x[data_col].to_numpy().reshape((nt, nt))
        return pd.DataFrame(
            {data_col: a2a.time_average(corr)}, index=pd.Index(range(nt), name=tvar)
        )

    df_out = group_apply(df, apply_func, data_col, list(avg_indices))

    return df_out


# def fold(df: pd.DataFrame, apply_fold: bool = True) -> pd.DataFrame:
#
#     if not apply_fold:
#         return df
#
#     assert len(df.columns) == 2
#
#     data_col = df.columns[-1]
#
#     array = df.sort_values('dt')[data_col].to_numpy()
#     nt = len(array)
#     folded_len = nt // 2 + 1
#     array[1:nt // 2] = (array[1:nt // 2] + array[-1:nt // 2:-1]) / 2.0
#
#     return pd.DataFrame(
#         array[:folded_len],
#         index=pd.Index(range(folded_len), name='dt'),
#         columns=[data_col]
#     )
#


def call(df, func_name, data_col, *args, **kwargs):
    func = globals().get(func_name, None)
    if callable(func):
        logging.debug(f"Calling {func_name} with args: {args}, kwargs: {kwargs}")
        return func(df, data_col, *args, **kwargs)
    else:
        raise AttributeError(f"Function '{func_name}' not found or is not callable.")


def execute(df: pd.DataFrame, actions: t.Dict) -> pd.DataFrame:
    df_out = df
    data_col = actions.pop("data_col", "corr")

    for key in sorted(actions.keys(), key=ACTION_ORDER.index):
        assert key in ACTION_ORDER
        param = actions[key]
        if isinstance(param, t.Dict):
            df_out = call(df_out, key, data_col, **param)
        elif isinstance(param, t.List):
            df_out = call(df_out, key, data_col, *param)
        else:
            if param:
                df_out = call(df_out, key, data_col, param)
            else:
                df_out = call(df_out, key, data_col)

    return df_out
