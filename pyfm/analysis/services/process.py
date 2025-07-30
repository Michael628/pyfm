import pandas as pd

from pyfm.analysis.components.mask import split_lmi_dsets
from pyfm.analysis.components.process import (
    calculate_lmi,
    calculate_high,
    average_gamma,
)
from pyfm.analysis.domain import BaseDataFrame


def average(df: pd.DataFrame) -> pd.DataFrame:
    return average_gamma(BaseDataFrame(df))


def append_lmi(df: pd.DataFrame) -> pd.DataFrame:
    ama, ranLL, a2aLL = split_lmi_dsets(df)
    print(ama, ranLL, a2aLL)
    high = calculate_high(ama, ranLL)
    lmi = calculate_lmi(high, a2aLL)

    return pd.concat([df, high, lmi])
