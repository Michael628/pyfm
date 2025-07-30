import dataclasses
from typing import Generic, Iterator, TypeVar, Callable, Any
import pandas as pd
from functools import reduce


# Any unprocessed data: ama, ranLL, a2aLL
class BaseDataFrame(pd.DataFrame):
    @property
    def _construct(self):
        return type(self)

    def __init__(self, *args, **kwargs):
        super(BaseDataFrame, self).__init__(*args, **kwargs)
        col_check = ["dset", "corr", "t", "series_cfg"]
        for col in col_check:
            assert col in self.columns, f"{col} not in data frame"
        assert len(self) != 0, "BaseDataFrame must not be empty"

        sort_by = ["series_cfg"]
        if self.has_gamma():
            sort_by += ["gamma"]
        if self.has_tsource():
            sort_by += ["tsource"]
        sort_by += ["t"]
        self.sort_values(sort_by, inplace=True)

    def has_tsource(self) -> bool:
        return "tsource" in self.columns

    def has_gamma(self) -> bool:
        return "gamma" in self.columns

    def get_dset(self) -> str:
        return self.iloc[0]["dset"]

    def get_nt(self) -> int:
        return self["t"].nunique()

    def assert_averaged(self) -> None:
        assert not self.has_tsource(), (
            f"{self.get_dset()} should not have 'tsource' column"
        )
        assert not self.has_gamma(), f"{self.get_dset()} should not have 'gamma' column"


# Processed Low mode projected random wall, "ranLL", data
class LowProjDataFrame(BaseDataFrame):
    def __init__(self, *args, **kwargs):
        super(LowProjDataFrame, self).__init__(*args, **kwargs)
        assert all(self["dset"] == "ranLL")


# Processed random wall, "ama", data
class LowSubRWDataFrame(BaseDataFrame):
    def __init__(self, *args, **kwargs):
        super(LowSubRWDataFrame, self).__init__(*args, **kwargs)
        assert all(self["dset"] == "high")


# Processed random wall, "ama", data
class RWDataFrame(BaseDataFrame):
    def __init__(self, *args, **kwargs):
        super(RWDataFrame, self).__init__(*args, **kwargs)
        assert all(self["dset"] == "ama")


# Processed low mode all-to-all, "a2aLL", data
class LowModeDataFrame(BaseDataFrame):
    def __init__(self, *args, **kwargs):
        super(LowModeDataFrame, self).__init__(*args, **kwargs)
        assert all(self["dset"] == "a2aLL")


# Processed Low mode averaged, "lmi", data
class LMIDataFrame(BaseDataFrame):
    def __init__(self, *args, **kwargs):
        super(LMIDataFrame, self).__init__(*args, **kwargs)
        assert all(self["dset"] == "lmi")


T = TypeVar("T")
ProcFn = Callable[[Any], Any]
FactoryFn = Callable[..., Iterator[T]]


@dataclasses.dataclass
class DataPipe(Generic[T]):
    _factory: FactoryFn
    _iter: Iterator[T]
    _callbacks: list[ProcFn]

    def __init__(self, factory: FactoryFn):
        self._factory = factory
        self.reset_iter()
        self.reset_callbacks()

    def reset_callbacks(self) -> None:
        self._callbacks = [lambda x: x]

    def reset_iter(self) -> None:
        self._iter = self._factory()

    def pipe(self, proc_fn: ProcFn):
        self._callbacks.append(proc_fn)
        return self

    def nop(self, proc_fn: ProcFn):
        def callback(df):
            proc_fn(df)
            return df

        self._callbacks.append(callback)
        return self

    def __next__(self):
        try:
            return reduce(lambda acc, f: f(acc), self._callbacks, next(self._iter))
        except StopIteration:
            self._iter = self._factory()
            raise StopIteration

    def __iter__(self):
        return self


class WrappedDataPipe(DataPipe):
    """
    Instead of passing the next data
    element to the stack directly, assumes that
    the data is wrapped in a tuple where the first element is a group descriptor,
    i.e. Iterator[tuple[group, data]]
    """

    def __next__(self):
        try:
            g, df = next(self._iter)
            return g, reduce(lambda acc, f: f(acc), self._callbacks, df)
        except StopIteration:
            self._iter = self._factory()
            raise StopIteration
