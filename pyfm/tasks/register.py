import typing as t
from pyfm.domain import HandlerRegistry, ConfigHandler, ConfigPreprocessorProtocol

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
    def default_preprocess_fn(params: t.Dict, subconfig: str | None = None) -> t.Dict:
        """Default preprocessing function that unwraps _tasks key to overwrite other params."""
        return params | params.get("_tasks", {})

    handler_key = get_task_key(config=config)

    handler = HandlerRegistry.register_config(handler_key, config)

    if len(funcs) > 0:
        HandlerRegistry.register_functions(handler_key, *funcs)

    for method_name, fn in kwfuncs.items():
        HandlerRegistry.register_function(handler_key, fn, method_name)

    # Provide default preprocessing function for tasks if not explicitly defined
    if not isinstance(handler, ConfigPreprocessorProtocol):
        HandlerRegistry.register_function(
            handler_key, default_preprocess_fn, "preprocess_params"
        )
