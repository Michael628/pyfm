import typing as t

from pyfm.domain import task_registry, build_hooks
from pyfm.domain.task_registry import TaskHandler
from pyfm import utils

# Mapping from positional function name to task_registry callable field name
_TASK_CALLABLE_NAMES = frozenset(
    {"build_input_params", "create_outfile_catalog", "build_aggregator_params"}
)

# Mapping from positional function name to build_hooks field name
_HOOK_NAME_MAP: dict[str, str] = {
    "preprocess_params": "preprocess",
    "postprocess_config": "postprocess",
}

# kwarg names that route to build_hooks
_KWARG_HOOK_MAP: dict[str, str] = {
    "preprocess_params": "preprocess",
    "postprocess_config": "postprocess",
    "validate": "validate",
}


def _default_preprocess(params: t.Dict) -> t.Dict:
    return params | params.pop("_preprocessor", {})


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
        raise ValueError("Must provide either `job_type` or `config` parameter.")

    return f"{scope}_{handler_key}"


def get_task_handler(
    job_type: str | None = None,
    task_type: str | None = None,
    config: t.Type | None = None,
    strict: bool = True,
) -> TaskHandler | None:
    from pyfm.domain.protocols import TaskHandlerProtocol

    handler_key = get_task_key(job_type, task_type, config)
    if handler_key is None:
        return None

    try:
        handler = task_registry.get(handler_key)
    except KeyError as e:
        utils.get_logger().debug(str(e))
        return None

    if strict and not isinstance(handler, TaskHandlerProtocol):
        utils.get_logger().debug(
            f"Handler '{handler_key}' does not satisfy TaskHandlerProtocol."
        )
        return None

    return handler


def list_registered_types() -> t.List[str]:
    return list(task_registry._handlers.keys())


def register_task(config: t.Type, *funcs, **kwfuncs) -> None:
    handler_key = get_task_key(config=config)
    if handler_key is None:
        utils.get_logger().debug(
            f"register_task called with config lacking a 'key' ClassVar: {config}"
        )
        return

    task_callables: dict[str, t.Callable] = {}
    hook_callables: dict[str, t.Callable] = {}

    # Route positional functions by their __name__
    for fn in funcs:
        name = getattr(fn, "__name__", None)
        if name in _TASK_CALLABLE_NAMES:
            task_callables[name] = fn
        elif name in _HOOK_NAME_MAP:
            hook_key = _HOOK_NAME_MAP[name]
            hook_callables[hook_key] = fn
        else:
            utils.get_logger().debug(
                f"register_task: ignoring positional function '{name}' "
                f"for key '{handler_key}' — not a recognised callable name."
            )

    # Route keyword functions
    for kw, fn in kwfuncs.items():
        if kw in _KWARG_HOOK_MAP:
            hook_callables[_KWARG_HOOK_MAP[kw]] = fn
        elif kw in _TASK_CALLABLE_NAMES:
            task_callables[kw] = fn
        else:
            utils.get_logger().debug(
                f"register_task: ignoring keyword '{kw}' for key '{handler_key}'."
            )

    # Register the TaskHandler (idempotent: skip if already registered)
    if handler_key not in task_registry._handlers:
        task_registry.register(handler_key, config, **task_callables)
    else:
        utils.get_logger().debug(
            f"register_task: handler '{handler_key}' already registered; skipping."
        )

    # Provide default preprocess hook when none explicitly supplied
    if "preprocess" not in hook_callables:
        hook_callables["preprocess"] = _default_preprocess

    # Register build hooks (idempotent: skip if already registered)
    if config not in build_hooks._registry:
        build_hooks.register(config, **hook_callables)
    else:
        utils.get_logger().debug(
            f"register_task: hooks for '{config.__name__}' already registered; skipping."
        )
