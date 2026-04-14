import typing as t
from pyfm.domain import HandlerRegistry, ConfigHandler

from pyfm import utils


def get_task_key(
    job_type: str | None = None,
    task_type: str | None = None,
    config: t.Type | None = None,
) -> str | None:
    scope = "nanny"
    if job_type is not None:
        handler_key = "_".join([job_type, task_type] if task_type else [job_type])
    elif config is not None:
        try:
            handler_key = config.key
        except AttributeError:
            utils.get_logger().debug(f"Config key not provided for: {config}")
            return None
    else:
        raise ValueError(f"Must provide either `job_type` or `config` parameter.")

    handler_key = HandlerRegistry.get_handler_key(scope, handler_key)

    return handler_key


def get_task_handler(
    job_type: str | None = None,
    task_type: str | None = None,
    config: t.Type | None = None,
) -> ConfigHandler | None:
    handler_key = get_task_key(job_type, task_type, config)

    try:
        return HandlerRegistry.get_handler(handler_key)
    except ValueError as e:
        utils.get_logger().debug(str(e))
        return None


def register_task(config: t.Type, *funcs, **kwfuncs):
    handler_key = get_task_key(config=config)

    HandlerRegistry.register_config(handler_key, config)
    if len(funcs) > 0:
        HandlerRegistry.register_functions(handler_key, *funcs)

    for method_name, fn in kwfuncs.items():
        HandlerRegistry.register_function(handler_key, fn, method_name)

    # Register default preprocessor if one has not been explicitly provided.
    # The default merges the _preprocessor slice (routed by the builder) into
    # the top-level params so child configs receive their params directly.
    handler = HandlerRegistry.get_handler(handler_key)
    if not hasattr(handler, "preprocess_params"):

        def _default_preprocessor(params: t.Dict) -> t.Dict:
            return params | params.get("_preprocessor", {})

        HandlerRegistry.register_function(
            handler_key, _default_preprocessor, "preprocess_params"
        )
