import typing as t
from pydantic.dataclasses import dataclass
from dataclasses import fields
from .protocols import FromDictProtocol


class HadronsInput(t.NamedTuple):
    modules: t.Dict[str, t.Dict]
    schedule: t.List[str]


@dataclass(frozen=True)
class MassDict:
    _items: t.Dict[str, float]

    @classmethod
    def from_dict(cls, kwargs) -> "MassDict":
        default = {"zero": 0.0}
        return cls(_items=default | kwargs)

    def __getitem__(self, key):
        return self._items[key]

    def keys(self):
        return self._items.keys()

    def values(self):
        return self._items.values()

    def items(self):
        return self._items.items()

    def to_string(self, mass_label: str, remove_prefix: bool = False) -> str:
        if remove_prefix:
            return str(self[mass_label]).removeprefix("0.")
        else:
            return str(self[mass_label])


@dataclass(frozen=True)
class LanczosParams(FromDictProtocol):
    alpha: float
    beta: float
    npoly: int
    nstop: int
    nk: int
    nm: int
    residual: float = 1e-8

    def keys(self):
        return [field.name for field in fields(self)]

    def __getitem__(self, key):
        return getattr(self, key)

    def values(self):
        return [getattr(self, k) for k in self.keys()]

    def items(self):
        return [(k, getattr(self, k)) for k in self.keys()]

    def to_string(self) -> t.Dict:
        return {k: str(v) for k, v in self.items()}
