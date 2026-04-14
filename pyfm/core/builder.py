import typing as t
from pyfm.domain import (
    ConfigBase,
    ConfigBuilder,
    CompositeConfig,
    SimpleConfig,
    build_hooks,
)

from pyfm import utils


def build_config(
    config_type,
    params: t.Dict[str, t.Any],
    file_params: t.Dict[str, t.Any] | None = None,
) -> ConfigBase:
    """Build a configuration object from input parameters.

    Hooks for *config_type* are looked up from the global ``build_hooks``
    registry and applied in the following order:

    1. Preprocessing: ``hooks.preprocess(params)`` — transform raw params
    2. Construction: Build config recursively (subconfigs built first)
    3. Postprocessing: ``hooks.postprocess(config)`` — transform built config
    4. Validation: ``hooks.validate(config)`` — assert invariants
    """

    if file_params is None:
        file_params = {}

    hooks = build_hooks.get(config_type)

    def preprocess(raw_params: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        if hooks is not None and hooks.preprocess is not None:
            return hooks.preprocess(raw_params)
        return raw_params

    def postproc_and_validate(config: ConfigBase) -> ConfigBase:
        if hooks is None:
            return config
        if hooks.postprocess is not None:
            config = hooks.postprocess(config)
        if hooks.validate is not None:
            hooks.validate(config)
        return config

    def new_builder(build_params: t.Dict[str, t.Any]) -> ConfigBuilder:
        return ConfigBuilder(config_type).with_yaml(build_params)

    def build_simple_config() -> SimpleConfig:
        processed_params = preprocess(params)
        config = new_builder(processed_params).with_files(file_params).build()
        return postproc_and_validate(config)

    def build_composite_config() -> CompositeConfig:
        """Return new CompositeConfig after recursively building all subconfigs."""

        processed_params = preprocess(params)

        subconfigs = {}
        for subconfig_label, field in config_type.get_subconfigs().items():
            match field.container:
                case field.container.SIMPLE:
                    sub_params = processed_params | {
                        "_preprocessor": processed_params.get("_preprocessor", {}).get(
                            subconfig_label, {}
                        )
                    }
                    subconfigs[subconfig_label] = build_config(
                        field.type, sub_params, file_params
                    )
                case field.container.LIST:
                    subconfig_key = subconfig_label.removesuffix("_config")
                    # Convert all params into list of params
                    if subconfig_label not in processed_params:
                        param_list = [processed_params]
                    elif not isinstance(processed_params[subconfig_label], list):
                        param_list = [
                            processed_params | processed_params[subconfig_label]
                        ]
                    else:
                        param_list = [
                            processed_params | sub_par
                            for sub_par in processed_params[subconfig_label]
                        ]

                    subconfigs[subconfig_label] = []
                    for sub_par in param_list:
                        routed = sub_par | {
                            "_preprocessor": processed_params.get(
                                "_preprocessor", {}
                            ).get(subconfig_label, {})
                        }
                        subconfigs[subconfig_label].append(
                            build_config(field.type, routed, file_params)
                        )

                case field.container.DICT:
                    param_provided = subconfig_label in processed_params and isinstance(
                        processed_params[subconfig_label], dict
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
