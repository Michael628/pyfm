import pandas as pd
from collections import namedtuple
from typing import NamedTuple, Iterator, Tuple

from ..components.mask import mask_incomplete_dsets, mask_outliers
from ..domain import BaseDataFrame, WrappedDataPipe


class SplitDataFrame(NamedTuple):
    passed: pd.DataFrame
    filtered: pd.DataFrame


def filter_outliers(df: pd.DataFrame, n_std=3, filter_by_cfg=True) -> SplitDataFrame:
    df_out = BaseDataFrame(df)

    mask = mask_outliers(df_out, n_std, filter_by_cfg)

    return SplitDataFrame(passed=df_out.loc[mask], filtered=df_out.loc[~mask])


def filter_incomplete_lmi(df: pd.DataFrame) -> SplitDataFrame:
    df_out = BaseDataFrame(df)
    assert set(df_out["dset"].unique()).issubset(["ama", "ranLL", "a2aLL"]), (
        f"Unexpected dset value: {df_out['dset'].unique()}"
    )
    assert df_out["dset"].nunique() == 3, "DataFrame must have 3 data sets"
    mask = mask_incomplete_dsets(df_out)

    return SplitDataFrame(passed=df_out.loc[mask], filtered=df_out.loc[~mask])


def create_col_iter(df: pd.DataFrame, group_cols: list[str]) -> WrappedDataPipe:
    def generate_column_masks() -> Iterator[Tuple[NamedTuple, pd.Series]]:
        """Yields a mask for each combination of columns in `cols`"""
        assert len(group_cols) > 0, "Must provide columns"
        assert set(group_cols).issubset(df.columns), (
            f"Not all columns ({group_cols}) in df ({df.columns})."
        )
        GroupTuple = namedtuple("GroupTuple", group_cols)
        groups = df[group_cols].assign(group_num=df.groupby(group_cols).ngroup())
        for i, (group, _) in enumerate(groups.groupby(group_cols)):
            yield (GroupTuple(*group), groups["group_num"] == i)

    def generate_column_dfs() -> Iterator[Tuple[NamedTuple, pd.DataFrame]]:
        for group, mask in generate_column_masks():
            yield group, df[mask]

    return WrappedDataPipe(generate_column_dfs)
