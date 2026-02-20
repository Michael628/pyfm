from pydantic.dataclasses import dataclass
from dataclasses import fields
from enum import Enum, auto
from typing import Dict, Any
import typing as t
import inspect
from pyfm import utils


class SerializableEnum(Enum):
    @classmethod
    def from_dict(cls, name: str) -> "SerializableEnum":
        if not isinstance(name, str):
            raise ValueError(
                f"Parameter passed to serializable type must be string, received: {name}"
            )
        name = name.upper().replace("_", "")
        if val := getattr(cls, name, None):
            return val
        raise ValueError(f"Invalid serializable type ({name}). options are: {list(cls)}")



@dataclass(frozen=True)
class ConfigBase:
    formatting: Dict
    logging_level: str
    runid: str

    def format_string(self, to_format: str) -> str:
        return to_format.format_map(self.formatting)


@dataclass(frozen=True)
class SimpleConfig(ConfigBase):
    pass


@dataclass(frozen=True)
class CompositeConfig(ConfigBase):
    @classmethod
    def get_subconfigs(cls) -> t.Dict[str, utils.ContainerType]:
        subconfigs = {}

        config_field_types = ((f.name, f.type) for f in fields(cls))
        subconfig_iter = utils.iterate_container(
            config_field_types,
            cond=lambda x: (inspect.isclass(x) and issubclass(x, ConfigBase)),
        )
        for field in subconfig_iter:
            subconfigs[field.name] = field

        return subconfigs
