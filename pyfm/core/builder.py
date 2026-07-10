import typing as t
from pyfm.core.configbuilder import ConfigBuilder
from pyfm.domain import (
    ConfigBase,
    CompositeConfig,
    SimpleConfig,
    build_hooks,
)


def build_config(
    config_type,
    params: t.Dict[str, t.Any],
    file_params: t.Dict[str, t.Any] | None = None,
    *,
    normalized: bool = False,
) -> ConfigBase:
    """Build a configuration object from input parameters.

    Hooks for *config_type* are looked up from the global ``build_hooks``
    registry and applied in the following order:

    1. Preprocessing — two ordered sub-steps:
       a. ``hooks.normalize(params)`` — *broad → canonical* transforms. Skipped
          when ``normalized=True`` (the input is already canonical, e.g. a
          generated task-input file).
       b. ``hooks.route(params)`` — ``_preprocessor`` routing. Always runs.
    2. Construction: Build config recursively (subconfigs built first)
    3. Postprocessing: ``hooks.postprocess(config)`` — transform built config
    4. Validation: ``hooks.validate(config)`` — assert invariants

    ``normalized`` propagates unchanged into every recursively-built subconfig,
    so the whole tree is treated as either raw or already-canonical.
    """

    if file_params is None:
        file_params = {}

    hooks = build_hooks.get(config_type)

    def preprocess(raw_params: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        if hooks is None:
            return raw_params
        if not normalized and hooks.normalize is not None:
            raw_params = hooks.normalize(raw_params)
        if hooks.route is not None:
            raw_params = hooks.route(raw_params)
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
        """Build a ``CompositeConfig`` by recursively constructing each subconfig field.

        After the composite type's ``route`` hook, ``_preprocessor`` is a nested
        routing table keyed by **exact** subconfig field names (the same keys as
        ``get_subconfigs()``).

        **SIMPLE:** ``_preprocessor[subconfig_label]`` is a dict (default ``{}``). That
        dict is passed as the child's ``_preprocessor`` when calling ``build_config``.

        **LIST:** ``_preprocessor[subconfig_label]`` must be a list (default ``[]`` if
        the key is missing). Each element is passed as the child's ``_preprocessor`` --
        the child's own route hook decides how to apply it. A non-list value
        raises ``TypeError``.

        **DICT:** ``processed_params[subconfig_label]`` maps child keys to per-key param
        dicts. ``_preprocessor[subconfig_label]`` is a dict mapping those same keys to
        per-child preprocessor dicts (default ``{}`` per key). Each child is built from
        ``processed_params | sub_params | {"_preprocessor": slice}``.
        """

        processed_params = preprocess(params)
        prep = processed_params.get("_preprocessor", {})

        subconfigs = {}
        for subconfig_label, field in config_type.get_subconfigs().items():
            match field.container:
                case field.container.SIMPLE:
                    slice_ = prep.get(subconfig_label, {})
                    if not isinstance(slice_, dict):
                        raise TypeError(
                            f"_preprocessor[{subconfig_label!r}] must be a dict for a "
                            f"SIMPLE subconfig, got {type(slice_).__name__}"
                        )
                    sub_params = processed_params | {"_preprocessor": slice_}
                    subconfigs[subconfig_label] = build_config(
                        field.type, sub_params, file_params, normalized=normalized
                    )
                case field.container.LIST:
                    raw_list = prep.get(subconfig_label, [])
                    if not isinstance(raw_list, list):
                        raise TypeError(
                            f"_preprocessor[{subconfig_label!r}] must be a list for a "
                            f"LIST subconfig, got {type(raw_list).__name__}"
                        )
                    subconfigs[subconfig_label] = []
                    for sub_par in raw_list:
                        if not isinstance(sub_par, dict):
                            raise TypeError(
                                f"Each entry of _preprocessor[{subconfig_label!r}] "
                                f"must be a dict, got {type(sub_par).__name__}"
                            )
                        routed = processed_params | {"_preprocessor": sub_par}
                        subconfigs[subconfig_label].append(
                            build_config(
                                field.type, routed, file_params, normalized=normalized
                            )
                        )

                case field.container.DICT:
                    key_params = processed_params.get(subconfig_label, {})
                    if not isinstance(key_params, dict):
                        raise TypeError(
                            f"params[{subconfig_label!r}] must be a dict for a "
                            f"DICT subconfig, got {type(key_params).__name__}"
                        )
                    key_slices = prep.get(subconfig_label, {})
                    if not isinstance(key_slices, dict):
                        raise TypeError(
                            f"_preprocessor[{subconfig_label!r}] must be a dict for a "
                            f"DICT subconfig, got {type(key_slices).__name__}"
                        )

                    subconfigs[subconfig_label] = {}
                    for key, sub_par in key_params.items():
                        if not isinstance(sub_par, dict):
                            raise TypeError(
                                f"params[{subconfig_label!r}][{key!r}] must be "
                                f"a dict, got {type(sub_par).__name__}"
                            )
                        slice_ = key_slices.get(key, {})
                        if not isinstance(slice_, dict):
                            raise TypeError(
                                f"_preprocessor[{subconfig_label!r}][{key!r}] must be "
                                f"a dict, got {type(slice_).__name__}"
                            )
                        subconfigs[subconfig_label][key] = build_config(
                            field.type,
                            processed_params | sub_par | {"_preprocessor": slice_},
                            file_params,
                            normalized=normalized,
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
