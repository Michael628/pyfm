from pydantic.dataclasses import dataclass
from dataclasses import fields
from enum import Enum, auto
from typing import Dict, Any
import typing as t
import inspect
from pyfm import utils


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
    def get_subconfigs(cls) -> utils.ContainerType:
        subconfigs = {}

        config_field_types = ((f.name, f.type) for f in fields(cls))
        subconfig_iter = utils.iterate_container(
            config_field_types,
            cond=lambda x: (inspect.isclass(x) and issubclass(x, ConfigBase)),
        )
        for field in subconfig_iter:
            subconfigs[field.name] = field

        return subconfigs
