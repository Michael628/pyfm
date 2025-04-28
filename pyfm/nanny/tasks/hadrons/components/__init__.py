from pydantic.dataclasses import dataclass
import typing as t
import pyfm


@dataclass
class ComponentBase(pyfm.ObserverInterface):
    @classmethod
    def from_dict(cls, kwargs: t.Dict[str, t.Any]) -> "ComponentBase":
        return cls(**kwargs)
