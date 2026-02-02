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
class ConfigPreprocessorProtocol(t.Protocol):
    def preprocess_params(self, params: t.Dict, subconfig: str | None = None) -> t.Dict:
        """Perform any necessary modifications to config input parameters before they
        are passed to the config constructor.
        """
        ...


@t.runtime_checkable
class ConfigPostprocessorProtocol(t.Protocol):
    def postprocess_config(self) -> "ConfigPostProcessorProtocol":
        """Perform any necessary modifications to subconfigs after they have been built."""
        ...


@t.runtime_checkable
class ConfigValidatorProtocol(t.Protocol):
    def validate(self) -> None:
        """Validate config after construction and postprocessing.

        This method is called as the final phase of config building, after all
        postprocessing has been applied. The validator function receives the config
        as the first parameter (auto-injected by ConfigHandler) and should raise
        exceptions if validation fails.
        """
        ...


@t.runtime_checkable
class TaskHandlerProtocol(t.Protocol):
    """A handler that can independently generate inputs for external programs.

    Any handler satisfying this protocol is complete and standalone.
    External code (scripts, CLI tools) should only use handlers satisfying this.

    Requires build_input_params, and create_outfile_catalog.
    """

    def build_input_params(self, config) -> t.Any:
        """Generate complete input with no external dependencies."""
        ...

    def create_outfile_catalog(self, config) -> t.Any:
        """List expected output files."""
        ...
