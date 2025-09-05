import typing as t
from dataclasses import Field, fields
from .outfiles import Outfile
from .protocols import FromDictProtocol, FormattableProtocol
from .conftypes import ConfigBase
from pyfm import utils

StringableTypes = t.Union[str, int, float, bool]


class ConfigBuilder:
    """Base builder for all configuration types.

    Attributes
    ----------
    _config_data : Dict[str, Any]
        The configuration parameters to pass to config constructor.
    _format_data : Dict[str, Any]
        Mappings for string formatting.
    _task_fields : List[str]
        List of dataclass field names defined in the task config class being built.
    """

    def __init__(self, task_class: ConfigBase):
        self._task_class = task_class
        self._config_data: t.Dict[str, t.Any] = {}
        self._format_data: t.Dict[str, t.Any] = {}
        self._task_fields: t.Dict[str, Field] = {
            f.name: f.type for f in fields(task_class)
        }

    def format_config_data(self):
        """Formats any string values in the config_data dictionary that contain format_data key replacements."""

        class PartialFormatter(dict):
            def __missing__(self, key):
                return "{" + key + "}"

        formatter = PartialFormatter(**self._format_data)
        for key, value in self._config_data.items():
            if isinstance(value, FormattableProtocol):
                self._config_data[key] = value.format_map(formatter)

        return self

    def build(self) -> ConfigBase:
        """Build the final Configuration."""
        missing = [k for k in self._task_fields.keys() if k not in self._config_data]
        if "formatting" in missing:
            self.with_field("formatting", self._format_data)
            missing.remove("formatting")
        if missing:
            utils.get_logger().warn(
                f"Missing fields for {self._task_class.__name__} Parameters: {missing}"
            )

        self.format_config_data()

        return self._task_class(**self._config_data)

    def with_field(self, key: str, value: t.Any):
        """If field `key` is a property of the config, set it to `value`.
        Otherwise, add as a formatting parameter.
        """

        if key in self._task_fields:
            task_type = utils.extract_non_none_type(self._task_fields[key])
            if utils.satisfies_protocol(task_type, FromDictProtocol):
                self._config_data[key] = task_type.from_dict(value)
            else:
                # Try to intelligently handle string params for list types
                if origin := getattr(task_type, "__origin__", None):
                    if origin == list:
                        if element_type := getattr(task_type, "__args__", [None])[0]:
                            try:
                                value = [element_type(value)]
                            except TypeError:
                                pass

                self._config_data[key] = value

        if isinstance(value, StringableTypes):
            self.with_formatter(key, value)
        else:
            utils.get_logger().debug(
                f"Not using `{key}` (type: {type(value).__name__}) for formatting: {value}"
            )

        return self

    def with_formatter(self, key: str, value: StringableTypes):
        if key in self._format_data:
            utils.get_logger().debug(
                f"Formatting {key} already in format_data with {self._format_data[key]}"
            )
            utils.get_logger().debug(f"Attempting to replace with {value}")

        self._format_data[key] = str(value)

        return self

    def with_files(self, files_config: t.Dict[str, t.Any]):
        file_path: str = files_config.get("home", "")
        for key, value in self._task_fields.items():
            if value is not Outfile:
                continue
            if key in files_config:
                self.with_field(
                    key, Outfile.from_param(key, file_path, files_config[key])
                )
            else:
                raise ValueError(f"Missing file parameters for {key}")

        return self

    def with_yaml(self, yaml_data: t.Dict[str, t.Any]):
        for key, value in yaml_data.items():
            self.with_field(key, value)

        return self

    def with_yaml_section(self, yaml_data: t.Dict[str, t.Any], section: str):
        return self.with_yaml(yaml_data.get(section, {}))
