import logging
import pandas as pd

from typing import Tuple
from ..domain import BaseDataFrame, LowModeDataFrame, LowProjDataFrame, RWDataFrame


def mask_outliers(df: BaseDataFrame, n_std: int, filter_by_cfg) -> BaseDataFrame:
    """Removes outlier points from df

    Args:
        df: BaseDataframe
        filter_by_cfg: If true, removes all points in df that share a config with an outlier

    Yields:
        NamedTuple describing group filter and filtered dataframe
    """

    mean = df.groupby("t")["corr"].transform("mean")
    std = df.groupby("t")["corr"].transform("std")
    cutoff = std * n_std
    lower, upper = mean - cutoff, mean + cutoff
    mask = (df["corr"] >= lower) & (df["corr"] <= upper)

    if filter_by_cfg:
        outlier_cfgs = df[~mask]["series_cfg"].drop_duplicates().to_list()
        if len(outlier_cfgs) != 0:
            logging.info(f"Outlier configs ({len(outlier_cfgs)}): {outlier_cfgs}")
        else:
            logging.debug(f"No outliers found.")

        return ~(df["series_cfg"].isin(outlier_cfgs))
    else:
        return mask


def mask_incomplete_dsets(
    df: BaseDataFrame, expected_dsets: int = 3, invert=False
) -> pd.Series:
    mask = df.groupby("series_cfg")["dset"].nunique() == expected_dsets
    # NOTE: how it's done when series_cfg is in index
    # mask = mask.loc[df.index.get_level_values("series_cfg")].values

    incomplete_configs = mask[~mask].index.values

    if len(incomplete_configs) != 0:
        logging.debug(
            f"Incomplete configs ({len(incomplete_configs)}): {incomplete_configs}"
        )
    else:
        logging.debug("`complete_dset_mask`: No configs filtered.")

    mask = mask.loc[df["series_cfg"]].values
    if len(df[mask]) == 0:
        raise ValueError(f"No configs found with {expected_dsets} data sets")

    if invert:
        mask = ~mask
    return mask


def split_lmi_dsets(
    df: BaseDataFrame,
) -> Tuple[RWDataFrame, LowProjDataFrame, LowModeDataFrame]:
    return (
        RWDataFrame(df.loc[df["dset"] == "ama"]),
        LowProjDataFrame(df.loc[df["dset"] == "ranLL"]),
        LowModeDataFrame(df.loc[df["dset"] == "a2aLL"]),
    )
