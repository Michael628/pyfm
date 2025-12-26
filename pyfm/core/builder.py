import typing as t
from pyfm.domain import (
    ConfigBase,
    ConfigBuilder,
    ConfigPreprocessorProtocol,
    CompositeConfig,
    SimpleConfig,
    HandlerRegistry,
    ConfigHandler,
)

from pyfm import utils


def build_config(
    config_type,
    config_params: t.Dict[str, t.Any],
    file_params: t.Dict[str, t.Any] | None = None,
    get_handler: t.Callable[[None], ConfigHandler] | None = None,
) -> ConfigBase:
    """Build a configuration object from input parameters."""

    if file_params is None:
        file_params = {}

    def preproc_fn(par, sub):
        if get_handler is not None:
            handler = get_handler(config_type)
            if isinstance(handler, ConfigPreprocessorProtocol):
                return handler.preprocess_params(par, subconfig=sub)
        return par

    def new_builder(build_params: t.Dict[str, t.Any]) -> ConfigBuilder:
        """Return new ConfigBuilder.
        Builder is loaded with `config_params` after wrapping with all preprocessors, if provided.
        """
        return ConfigBuilder(config_type).with_yaml(build_params)

    def build_simple_config() -> SimpleConfig:
        processed_params = preproc_fn(config_params, None)

        return new_builder(processed_params).with_files(file_params).build()

    def build_composite_config() -> CompositeConfig:
        """Return new CompositeConfig after recursively building all subconfigs."""

        processed_params = preproc_fn(config_params, None)

        subconfigs = {}
        for subconfig_label, field in config_type.get_subconfigs().items():

            # Remove "_config" suffix to get clean key
            subconfig_key = subconfig_label.removesuffix("_config")

            # Preprocess for subconfig
            processed_sub_params = preproc_fn(processed_params, subconfig_key)

            match field.container:
                case field.container.SIMPLE:
                    subconfigs[field.name] = build_config(
                        field.type, processed_sub_params, file_params, get_handler
                    )
                case field.container.LIST:
                    # Convert all params into list of params
                    if field.name not in processed_sub_params:
                        param_list = [processed_sub_params]
                    elif not isinstance(processed_sub_params[field.name], list):
                        param_list = [
                            processed_sub_params | processed_sub_params[field.name]
                        ]
                    else:
                        param_list = [
                            processed_sub_params | sub_par
                            for sub_par in processed_sub_params[field.name]
                        ]

                    subconfigs[field.name] = []
                    for sub_par in param_list:
                        subconfigs[field.name].append(
                            build_config(field.type, sub_par, file_params, get_handler)
                        )

                case field.container.DICT:
                    param_provided = (
                        subconfig_label in processed_sub_params
                        and isinstance(processed_sub_params[subconfig_label], dict)
                    )
                    if not param_provided:
                        raise ValueError(
                            f"Expected key {subconfig_label} not found in params."
                        )

                    subconfigs[subconfig_label] = {}
                    for key, subconfig_params in processed_sub_params[
                        subconfig_label
                    ].items():
                        subconfigs[subconfig_label][key] = build_config(
                            field.type,
                            processed_sub_params | subconfig_params,
                            file_params,
                            get_handler,
                        )

        return (
            new_builder(processed_params | subconfigs).with_files(file_params).build()
        )

    if issubclass(config_type, CompositeConfig):
        return build_composite_config()
    elif issubclass(config_type, SimpleConfig):
        return build_simple_config()
    else:
        raise ValueError(f"Attempting to build invalid config type: {config_type}")


__all__ = [
    "build_config",
]
