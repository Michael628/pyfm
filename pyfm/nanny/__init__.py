import typing as t

from pydantic.dataclasses import dataclass
import pyfm


class TaskBase:
    @classmethod
    def from_dict(cls, kwargs: t.Dict[str, t.Any]) -> "TaskBase":
        return cls(**kwargs)


@dataclass
class SubmitConfig(pyfm.ConfigBase):
    ens: str
    time: int
