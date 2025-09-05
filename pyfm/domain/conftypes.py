from pydantic.dataclasses import dataclass
from dataclasses import fields
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
    def format_string(self, to_format: str) -> str:
        """Format the string using each format_string function in the subconfigs."""
        formatted_string = to_format
        for key in self.get_subconfigs().keys():
            formatted_string = getattr(self, key).format_string(formatted_string)
        return formatted_string

    @classmethod
    def get_subconfigs(cls) -> t.Dict[str, Any]:
        subconfigs = {}

        for field in fields(cls):
            field_type = utils.extract_non_none_type(field.type)

            try:
                if inspect.isclass(field_type) and issubclass(field_type, SimpleConfig):
                    subconfigs[field.name] = field_type
            except TypeError:
                pass

        return subconfigs
