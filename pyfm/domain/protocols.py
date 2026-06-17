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
class ConfigNormalizerProtocol(t.Protocol):
    def normalize_params(self, params: t.Dict) -> t.Dict:
        """Transform broad input parameters into canonical form before routing.

        Skipped when the input is already canonical (``normalized=True``).
        """
        ...


@t.runtime_checkable
class ConfigRouterProtocol(t.Protocol):
    def route_params(self, params: t.Dict) -> t.Dict:
        """Absorb the incoming ``_preprocessor`` slice and emit the outgoing one
        for child subconfigs. Always runs.
        """
        ...


@t.runtime_checkable
class ConfigPostprocessorProtocol(t.Protocol):
    def postprocess_config(self) -> "ConfigPostprocessorProtocol":
        """Perform any necessary modifications to subconfigs after they have been built."""
        ...


@t.runtime_checkable
class ConfigValidatorProtocol(t.Protocol):
    def validate(self) -> None:
        """Validate config after construction and postprocessing."""
        ...


# ---------------------------------------------------------------------------
# Task-handler protocols — config is an explicit first parameter in all methods
# ---------------------------------------------------------------------------

@t.runtime_checkable
class InputBuilderProtocol(t.Protocol):
    """Handler that can generate input parameters for an external program."""

    def build_input_params(self, config: t.Any) -> t.Any:
        """Generate input parameters; *config* is passed explicitly by the caller."""
        ...


@t.runtime_checkable
class OutfileCatalogProtocol(t.Protocol):
    """Handler that can enumerate expected output files."""

    def create_outfile_catalog(self, config: t.Any) -> t.Any:
        """Return a catalog of expected output files for *config*."""
        ...


@t.runtime_checkable
class AggregatorProtocol(t.Protocol):
    """Handler that can supply aggregation parameters."""

    def build_aggregator_params(self, config: t.Any) -> t.Any:
        """Return aggregation parameters for *config*."""
        ...


@t.runtime_checkable
class TaskHandlerProtocol(InputBuilderProtocol, OutfileCatalogProtocol, t.Protocol):
    """Composite protocol: a fully standalone task handler.

    Any handler satisfying this protocol has both ``build_input_params`` and
    ``create_outfile_catalog``.  External code (scripts, CLI tools) should
    only consume handlers that satisfy this protocol.
    """
