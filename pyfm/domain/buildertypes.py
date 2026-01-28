import typing as t

from dataclasses import fields
from pyfm.domain.outfiles import Outfile
from pyfm.domain.protocols import FromDictProtocol, FormattableProtocol
from pyfm.domain.conftypes import ConfigBase
from pyfm import utils
from pyfm.utils.string import PartialFormatter

StringableTypes = t.Union[str, int, float, bool]


class ConfigBuilder:
    """Base builder for all configuration types.

    Attributes
    ----------
    _input_params : Dict[str, Any]
        The configuration parameters to pass to config constructor.
    _format_params : Dict[str, Any]
        Mappings for string formatting.
    _config_fields : List[str]
        List of dataclass field names defined in the config class being built.
    """

    def __init__(self, config_class: t.Any):
        self._config_class = config_class
        self._input_params: t.Dict[str, t.Any] = {}
        self._format_params: t.Dict[str, t.Any] = {}
        self._config_fields: t.Dict[str, t.Any] = {
            f.name: f.type for f in fields(config_class)
        }

    def format_input_params(self):
        """Formats any string values in the config_data dictionary that contain format_data key replacements."""

        formatter = PartialFormatter(**self._format_params)
        iterator = utils.iterate_container(
            self._config_fields.items(),
            cond=lambda x: isinstance(x, FormattableProtocol),
        )
        for field in iterator:
            if field.name not in self._input_params:
                continue
            value = self._input_params[field.name]
            try:
                match field.container:
                    case field.container.SIMPLE:
                        value = value.format_map(formatter)
                    case field.container.LIST:
                        value = [v.format_map(formatter) for v in value]
                    case field.container.DICT:
                        value = {k: v.format_map(formatter) for k, v in value.items()}
                    case _:
                        continue
            except KeyError as e:
                raise ValueError(f"Couldn't find key in parameters: {e}")
            self._input_params[field.name] = value

        return self

    def build(self) -> ConfigBase:
        """Build the final Configuration."""
        missing = [k for k in self._config_fields.keys() if k not in self._input_params]
        if "formatting" in missing:
            self.with_field("formatting", self._format_params)
            missing.remove("formatting")
        if missing:
            utils.get_logger().debug(
                f"Missing fields for {self._config_class.__name__} Parameters: {missing}"
            )

        self.format_input_params()

        return self._config_class(**self._input_params)

    def with_field(self, key: str, value: t.Any):
        """If field `key` is a property of the config, set it to `value`.
        Otherwise, add as a formatting parameter.
        """

        if key in self._config_fields:
            container_type = utils.get_container(self._config_fields[key])

            def convert(raw):
                if isinstance(raw, container_type.type):
                    return raw
                elif utils.satisfies_protocol(container_type.type, FromDictProtocol):
                    return container_type.type.from_dict(raw)
                elif isinstance(raw, dict):
                    return container_type.type(**raw)
                else:
                    return raw

            match container_type.container:
                case container_type.container.SIMPLE:
                    try:
                        value = convert(value)
                    except TypeError:
                        pass
                case container_type.container.LIST:
                    try:
                        if isinstance(value, list):
                            value = [convert(v) for v in value]
                        else:
                            value = [convert(value)]
                    except TypeError:
                        pass
                case container_type.container.DICT:
                    try:
                        value = {k: convert(v) for k, v in value.items()}
                    except TypeError:
                        pass

            self._input_params[key] = value

        if isinstance(value, StringableTypes):
            self.with_formatter(key, value)
        else:
            utils.get_logger().debug(
                f"Not using `{key}` (type: {type(value).__name__}) for formatting: {value}"
            )

        return self

    def with_formatter(self, key: str, value: StringableTypes):
        if key in self._format_params:
            utils.get_logger().debug(
                f"Formatting {key} already in format_data with {self._format_params[key]}"
            )
            utils.get_logger().debug(f"Attempting to replace with {value}")

        self._format_params[key] = str(value)

        return self

    def iterate_outfiles(self) -> t.Iterable:
        """Iterates over properties of `_config_class` and yields results containing type Outfile
        (including dictionaries and lists"""
        yield from utils.iterate_container(
            self._config_fields.items(), cond=lambda x: x is Outfile
        )

    def with_files(self, files_config: t.Dict[str, t.Any]):
        def find_file_label(field_name) -> t.Iterator[str | None]:
            """Tries to find appropriate file parameter label. If `_task_fields[field_name]` contains a string already
            it will try using that value as a key in files_config. Otherwise it will try `file_name` itself.
            """

            match field_value := self._input_params.get(field_name, None):
                case str():
                    if field_value in files_config:
                        yield field_value
                    else:
                        utils.get_logger().debug(
                            f"Unexpected parameter found when searching for Outfile param ({field_name}): {field_value}"
                        )
                        yield None
                case list():
                    for l in field_value:
                        yield from find_file_label(l)
                case _:
                    match field_name:
                        case str():
                            if field_name in files_config:
                                yield field_name
                            else:
                                utils.get_logger().debug(
                                    f"Missing Outfile parameters for {field_name}"
                                )
                                yield None

                        case _:
                            raise ValueError(
                                f"Unexpected parameter found when searching for Outfile param: {field_name}"
                            )

        file_path: str = files_config.get("home", "")
        for field in self.iterate_outfiles():
            match field.container:
                case field.container.SIMPLE:
                    if label := next(find_file_label(field.name)):
                        self.with_field(
                            field.name,
                            Outfile.from_param(label, file_path, files_config[label]),
                        )
                case field.container.LIST:
                    vals = [
                        Outfile.from_param(l, file_path, files_config[l])
                        for l in find_file_label(field.name)
                        if l is not None
                    ]
                    if len(vals) > 0:
                        self.with_field(field.name, vals)
                case _:
                    raise NotImplementedError(
                        f"Outfile field container: {field.container} not implemented."
                    )

        return self

    def with_yaml(self, yaml_data: t.Dict[str, t.Any]):
        for key, value in yaml_data.items():
            self.with_field(key, value)

        return self

    def with_yaml_section(self, yaml_data: t.Dict[str, t.Any], section: str):
        return self.with_yaml(yaml_data.get(section, {}))
