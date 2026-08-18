import typing as t
from collections import namedtuple
import gvar as gv
import gvar.dataset as ds
import numpy as np
import pandas as pd
from pyfm import a2a, utils

ACTION_ORDER = [
    "preprocess_custom",
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
    "custom",
    "postprocess_custom",
]

MaskGroup = t.Tuple[t.Optional[t.NamedTuple], pd.Series]
MaskedDFGroup = t.Tuple[t.Optional[t.NamedTuple], pd.DataFrame]
BufferTuple = namedtuple("BufferTuple", ["columns", "buffer"])


def custom(df: pd.DataFrame, data_col: str, fn: t.Callable, *args, **kwargs):
    return fn(df, data_col, *args, **kwargs)


# Do the same thing regardless of preprocessing or postprocessing, only difference is order
preprocess_custom = postprocess_custom = custom


def norm_dist(df: pd.DataFrame) -> pd.DataFrame:
    def normalize(df: pd.DataFrame) -> pd.Series:
        return (
            df["corr"].sub(df["corr"].groupby("t").transform("mean"))
            if len(df) == 1
            else df["corr"]
            .sub(df["corr"].groupby("t").transform("mean"))
            .divide(df["corr"].groupby("t").transform("std"))
        )

    return df.set_index("t").assign(corr=normalize).reset_index("t")
    # return pd.concat([normalize(f) for _, f in df.groupby("t")])


def generate_column_masks(
    df: pd.DataFrame, group_cols: list[str] | None = None
) -> t.Generator[MaskGroup, None, None]:
    """Yields a mask for each combination of columns in `group_cols`"""
    if not group_cols:
        yield (None, pd.Series(True, index=df.index))
    else:
        GroupTuple = utils.create_group_tuple(*group_cols)
        groups = df[group_cols].assign(group_num=df.groupby(group_cols).ngroup())
        for i, (group, _) in enumerate(groups.groupby(group_cols)):
            yield (GroupTuple(*group), groups["group_num"] == i)


def generate_dfs(
    df: pd.DataFrame, mask: pd.Series | t.Iterator[MaskGroup] | None = None
) -> t.Generator[MaskedDFGroup, None, None]:
    if mask is None:
        yield None, df
    elif isinstance(mask, pd.Series):
        yield None, df[mask]
    else:
        for g, m in mask:
            yield g, df[m]


def generate_column_dfs(df, cols):
    yield from generate_dfs(df, generate_column_masks(df, cols))


def group_apply(
    df: pd.DataFrame,
    apply_fn: t.Callable,
    data_col: str,
    ungrouped_cols: t.List,
    invert: bool = False,
) -> pd.DataFrame:
    """Applies `apply_fn` to `data_col` in `df` grouped by `ungrouped_cols`.

    Parameters
    ----------
    df : pd.DataFrame

    apply_fn : t.Callable

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

    df_out = (
        df.reset_index().groupby(by=grouped, dropna=False)[ungrouped].apply(apply_fn)
    )

    return df_out


def drop(df, _: str, *args):
    for key in args:
        assert isinstance(key, str)

        if key in df.index.names:
            df.reset_index(key, drop=True, inplace=True)
        elif key in df.columns:
            _ = df.pop(key)
        else:
            utils.get_logger().debug(f"Drop request skipped. Key not found: {key}")
    return df


def index(df, _: str, *args) -> pd.DataFrame:

    indices = list(args)

    assert all([isinstance(i, str) for i in indices])

    if not indices:
        return df

    logger = utils.get_logger()
    series_cfg = "series_cfg"
    if "series.cfg" in indices:
        i = indices.index("series.cfg")
        indices[i] = series_cfg
    df.rename_axis(index={"series.cfg": series_cfg}, inplace=True)
    df.rename({"series.cfg": series_cfg}, inplace=True)
    logger.debug(f"Current df index: {df.index.names}")
    logger.debug(f"Current df columns: {df.columns}")
    logger.debug(f"Setting index as {indices}")

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


def normalize(df, _: str, divisor):
    return df["corr"].apply(lambda x: x / float(divisor)).to_frame()


def sum(df: pd.DataFrame, data_col, *sum_indices) -> pd.DataFrame:
    """Sums `data_col` column in `df` over columns or indices specified in `avg_indices`"""
    return group_apply(df, lambda x: x[data_col].mean(), data_col, list(sum_indices))


def average(df: pd.DataFrame, data_col, *avg_indices) -> pd.DataFrame:

    df_out = df
    final_columns = [n for n in df.columns if n not in avg_indices and n != data_col]
    for col in avg_indices:
        # Move all non-data columns to the index to avoid expensive reset_index()
        other_cols = [x for x in df_out.columns if x != data_col]
        cols_to_set = [c for c in other_cols if c not in df_out.index.names]
        if cols_to_set:
            df_out = df_out.set_index(cols_to_set, append=True)

        # Group by all index levels except the one being averaged
        unnamed = [i for i, name in enumerate(df_out.index.names) if name is None]
        if unnamed:
            utils.get_logger().warning(
                f"Excluding unnamed index level(s) at position(s) {unnamed} "
                f"from average groupby keys for '{col}'"
            )
        levels_to_keep = [
            name for name in df_out.index.names if name not in [None, col]
        ]
        df_out = df_out.groupby(level=levels_to_keep).mean()

    return df_out if len(final_columns) == 0 else df_out.reset_index(final_columns)


def permkey_split_old(
    df: pd.DataFrame, _: str, permkey_col: str = "permkey"
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

    If the data already carries a time-averaged ``t`` column (as an index level
    or a regular column), the time-average operation is skipped and the input
    DataFrame is returned unchanged. This lets the contract ``--average`` path
    consume data that has already been time-averaged instead of assuming the
    ``t1``/``t2`` columns used by the raw two-point array still exist.
    """
    if "dt" in df.index.names or "dt" in df.columns:
        df = df.rename_axis(
            index={"dt": "t"} if "dt" in df.index.names else None
        ).rename(columns={"dt": "t"})

    # Rename None-named index levels whose unique values are sequential
    # integers 0..n-1 to 't'.  This catches pre-averaged data from
    # mixed-naming concatenations (some files had 't', others 'dt' —
    # pandas resolves the conflict as None).
    if "t" not in df.columns:
        names = list(df.index.names)
        renamed = False
        for i, name in enumerate(names):
            if name is None:
                unique_vals = df.index.get_level_values(i).unique()
                n = len(unique_vals)
                if n > 0 and list(unique_vals) == list(range(n)):
                    names[i] = "t"
                    renamed = True
                    break
        if renamed:
            df.index = df.index.set_names(names)

    if "t" in df.index.names or "t" in df.columns:
        utils.get_logger().debug(
            "time_average skipped: 't' column already present in data"
        )
        return df

    assert len(avg_indices) == 2
    tvar = "t"

    def apply_fn(x):
        nt = int(np.sqrt(len(x)))
        assert nt**2 == len(x)
        corr = x[data_col].to_numpy().reshape((nt, nt))
        return pd.DataFrame(
            {data_col: a2a.time_average(corr)}, index=pd.Index(range(nt), name=tvar)
        )

    df_out = group_apply(df, apply_fn, data_col, list(avg_indices))

    # Normalize innermost index name to 't' — handles pandas groupby().apply()
    # silently dropping the name to None (version-dependent) or legacy 'dt'.
    # Position-agnostic: works for both flat Index and MultiIndex.
    if df_out.index.names[-1] != "t":
        names = list(df_out.index.names)
        names[-1] = "t"
        df_out.index = df_out.index.set_names(names)

    return df_out


def call(df, fn_name, data_col, *args, **kwargs):
    fn = globals().get(fn_name, None)
    if callable(fn):
        utils.get_logger().debug(
            f"Calling {fn_name} with args: {args}, kwargs: {kwargs}"
        )
        return fn(df, data_col, *args, **kwargs)
    else:
        raise AttributeError(f"Function '{fn_name}' not found or is not callable.")


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
