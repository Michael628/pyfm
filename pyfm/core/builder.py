import typing as t
from pyfm.domain import (
    ConfigBase,
    ConfigBuilder,
    ConfigPreprocessorProtocol,
    CompositeConfig,
    SimpleConfig,
    HandlerRegistry,
    ConfigHandler,
    ConfigPostprocessorProtocol,
    ConfigValidatorProtocol,
)

from pyfm import utils


def build_config(
    config_type,
    config_params: t.Dict[str, t.Any],
    file_params: t.Dict[str, t.Any] | None = None,
    get_handler: t.Callable[[None], ConfigHandler] | None = None,
) -> ConfigBase:
    """Build a configuration object from input parameters.

    Execution phases:
    1. Preprocessing: Call preprocess_params if handler provides it
    2. Construction: Build config recursively (subconfigs built first)
    3. Postprocessing: Call postprocess_config if handler provides it
    4. Validation: Call validate if config implements ValidatableConfig
    """

    if file_params is None:
        file_params = {}

    def preproc_fn(par):
        if get_handler is not None:
            handler = get_handler(config_type)
            if isinstance(handler, ConfigPreprocessorProtocol):
                return handler.preprocess_params(par)
        return par

    def postproc_and_validate(config: ConfigBase) -> ConfigBase:
        """Apply postprocessing and validation to a built config."""
        if get_handler is None:
            return config

        handler = get_handler(config_type)
        handler.config = config

        # Phase 3: Postprocessing
        if isinstance(handler, ConfigPostprocessorProtocol):
            handler.config = handler.postprocess_config()

        # Phase 4: Validation
        if isinstance(handler, ConfigValidatorProtocol):
            handler.validate()

        return handler.config

    def new_builder(build_params: t.Dict[str, t.Any]) -> ConfigBuilder:
        """Return new ConfigBuilder.
        Builder is loaded with `config_params` after wrapping with all preprocessors, if provided.
        """
        return ConfigBuilder(config_type).with_yaml(build_params)

    def build_simple_config() -> SimpleConfig:
        processed_params = preproc_fn(config_params, None)
        config = new_builder(processed_params).with_files(file_params).build()
        return postproc_and_validate(config)

    def build_composite_config() -> CompositeConfig:
        """Return new CompositeConfig after recursively building all subconfigs."""

        processed_params = preproc_fn(config_params)

        subconfigs = {}
        for subconfig_label, field in config_type.get_subconfigs().items():

            # Remove "_config" suffix to get clean key
            subconfig_key = subconfig_label.removesuffix("_config")

            # Preprocess for subconfig
            processed_sub_params = preproc_fn(processed_params, subconfig_key)

            match field.container:
                case field.container.SIMPLE:
                    sub_params = processed_params | {
                        "_preprocessor": processed_params.get("_preprocessor", {}).get(
                            subconfig_key, {}
                        )
                    }
                    subconfigs[field.name] = build_config(
                        field.type, sub_params, file_params, get_handler
                    )
                case field.container.LIST:
                    preprocessor_slice = processed_params.get(
                        "_preprocessor", {}
                    ).get(subconfig_key, {})
                    # Convert all params into list of params
                    if field.name not in processed_params:
                        param_list = [
                            processed_params | {"_preprocessor": preprocessor_slice}
                        ]
                    elif not isinstance(processed_params[field.name], list):
                        param_list = [
                            processed_params
                            | processed_params[field.name]
                            | {"_preprocessor": preprocessor_slice}
                        ]
                    else:
                        param_list = [
                            processed_params
                            | sub_par
                            | {"_preprocessor": preprocessor_slice}
                            for sub_par in processed_params[field.name]
                        ]

                    subconfigs[field.name] = []
                    for sub_par in param_list:
                        subconfigs[field.name].append(
                            build_config(field.type, sub_par, file_params, get_handler)
                        )

                case field.container.DICT:
                    param_provided = (
                        subconfig_label in processed_params
                        and isinstance(processed_params[subconfig_label], dict)
                    )
                    if not param_provided:
                        raise ValueError(
                            f"Expected key {subconfig_label} not found in params."
                        )

                    subconfigs[subconfig_label] = {}
                    for key, subconfig_params in processed_params[
                        subconfig_label
                    ].items():
                        subconfigs[subconfig_label][key] = build_config(
                            field.type,
                            processed_params | subconfig_params,
                            file_params,
                            get_handler,
                        )

        config = (
            new_builder(processed_params | subconfigs).with_files(file_params).build()
        )
        return postproc_and_validate(config)

    if issubclass(config_type, CompositeConfig):
        return build_composite_config()
    elif issubclass(config_type, SimpleConfig):
        return build_simple_config()
    else:
        raise ValueError(f"Attempting to build invalid config type: {config_type}")


__all__ = [
    "build_config",
]
