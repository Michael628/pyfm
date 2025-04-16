import typing as t

from pydantic.dataclasses import dataclass
import pyfm
from pyfm.nanny.config import Outfile


class TaskBase:
    @classmethod
    def from_dict(cls, kwargs: t.Dict[str, t.Any]) -> "TaskBase":
        return cls(**kwargs)


@dataclass
class SubmitConfig(pyfm.ConfigBase):
    ens: str
    time: int
    files: t.Dict[str, Outfile]
