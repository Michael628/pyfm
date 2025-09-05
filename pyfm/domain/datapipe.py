import dataclasses
from typing import Generic, Iterator, TypeVar, Callable, Any
import pandas as pd
from functools import reduce


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
        self.agg_fn = self.default_agg_fn

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

    def set_agg_fn(self, agg_fn: Callable[[], pd.DataFrame]):
        self.agg_fn = agg_fn
        return self

    def chain(self, other):
        assert isinstance(self, type(other))

        def factory():
            yield from self
            yield from other

        return type(self)(factory)

    @staticmethod
    def default_agg_fn(*args):
        if args:
            assert all([isinstance(a, pd.DataFrame) for a in args])
        return pd.concat(args)

    def agg(self):
        return self.agg_fn(*[df for df in self])


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

    def agg(self):
        return self.agg_fn(*[df for _, df in self])
