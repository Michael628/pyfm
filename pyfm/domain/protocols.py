import typing as t


@t.runtime_checkable
class FromDictProtocol(t.Protocol):
    @classmethod
    def from_dict(cls, kwargs: t.Dict[str, t.Any]) -> "FromDictProtocol":
        """Creates a new instance of the class from a dictionary."""
        return cls(**kwargs)


@t.runtime_checkable
class FormattableProtocol(t.Protocol):
    def format_map(self, mapping: t.Dict) -> t.Any:
        """Formats object contents according to replacements in `mapping`."""
        ...


@t.runtime_checkable
class ConfigProcessorProtocol(t.Protocol):
    def preprocess_params(self, params: t.Dict, subconfig: str | None = None) -> t.Dict:
        """Perform any necessary modifications to config input parameters before they
        are passed to the config constructor.
        """
        ...

    def postprocess_config(self) -> "ConfigProcessorProtocol":
        """Perform any necessary modifications to subconfigs after they have been built."""
        pass
