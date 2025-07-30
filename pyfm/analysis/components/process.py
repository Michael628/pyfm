import pyfm.processing.processor as pc
from ..domain import (
    BaseDataFrame,
    LMIDataFrame,
    LowModeDataFrame,
    LowProjDataFrame,
    LowSubRWDataFrame,
    RWDataFrame,
)


def average_gamma(df: BaseDataFrame) -> BaseDataFrame:
    return BaseDataFrame(pc.average(df, "corr", "gamma"))


def check_valid(*args: BaseDataFrame) -> None:
    assert len(args) > 0, "No arguments passed to validator"

    for df in args:
        assert not df.has_tsource(), f"{df.get_dset()} should not have 'tsource' column"
        assert df.has_gamma(), (
            f"{df.get_dset()} must have 'gamma' column for correlated difference of low projected noise."
        )
    lengths = list(map(len, args))
    assert all([len_df == lengths[0] for len_df in lengths]), (
        f"Mismatched argument lengths: {lengths}"
    )


def calculate_high(ama: RWDataFrame, ranLL: LowProjDataFrame) -> LowSubRWDataFrame:
    check_valid(ama, ranLL)

    return LowSubRWDataFrame(
        ama.copy()
        .assign(corr=ama["corr"].values - ranLL["corr"].values)
        .assign(dset="high")
    )


def calculate_lmi(high: LowSubRWDataFrame, a2aLL: LowModeDataFrame) -> LMIDataFrame:
    check_valid(high, a2aLL)

    return LMIDataFrame(
        high.copy()
        .assign(corr=(high["corr"].values + a2aLL["corr"].values))
        .assign(dset="lmi")
    )


BufferTuple = namedtuple("BufferTuple", ["columns", "buffer"])


def create_gvar_buffer(
    df: BaseDataFrame,
    key_index: t.Union[str, t.List[str]],
    fold: bool = False,
) -> BufferTuple:
    """
    Converts a pandas DataFrame into a grouped buffer dictionary.

    Parameters:
        df (pd.DataFrame): The input DataFrame, expected to have a multi-index with a time variable.
        key_index (Union[str, List[str]]): The column(s) or index level(s) to group by.

    Returns:
        gv.BufferDict: A dictionary-like object where keys are group identifiers and values are
                       numpy arrays of reshaped data corresponding to each group.

    Raises:
        AssertionError: If the specified key_index is not found in the DataFrame's columns or index levels.
    """

    df.assert_averaged()

    buff = gv.BufferDict()

    nt = df.get_nt()

    if isinstance(key_index, str):
        key_indices = [key_index]
    else:
        key_indices = key_index

    buff_fold = gv.BufferDict()

    for key, xs in df.groupby(key_indices, as_index=False):
        k = key if len(key_indices) > 1 else key[0]
        buff[k] = xs["corr"].to_numpy().reshape((-1, nt))
        if fold:
            size = buff[k].shape
            size = (size[0], size[1] // 2 + 1)

            buff_fold[k] = np.zeros(size, dtype=buff[k].dtype)

            buff_fold[k][:, 1 : nt // 2] = (
                buff[k][:, 1 : nt // 2] + buff[k][:, -1 : nt // 2 : -1]
            ) / 2.0
            buff_fold[k][:, 0] = buff[k][:, 0]
            buff_fold[k][:, nt // 2] = buff[k][:, nt // 2]

    return BufferTuple(buffer=buff if not fold else buff_fold, columns=key_indices)
