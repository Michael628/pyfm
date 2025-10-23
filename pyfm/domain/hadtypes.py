import typing as t
from pydantic.dataclasses import dataclass
from dataclasses import fields
from pyfm.domain.protocols import FromDictProtocol


class HadronsInput(t.NamedTuple):
    modules: t.Dict[str, t.Dict]
    schedule: t.List[str]


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
