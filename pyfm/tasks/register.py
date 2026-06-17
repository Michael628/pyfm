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
    "normalize_params": "normalize",
    "route_params": "route",
    "postprocess_config": "postprocess",
}

# kwarg names that route to build_hooks
_KWARG_HOOK_MAP: dict[str, str] = {
    "normalize_params": "normalize",
    "route_params": "route",
    "postprocess_config": "postprocess",
    "validate": "validate",
}

# Reverse mapping: config class → task key (without "nanny_" prefix), populated by register_task
_config_to_task_key: dict[t.Type, str] = {}


def _default_route(params: t.Dict) -> t.Dict:
    """Default ``route`` hook: absorb the incoming ``_preprocessor`` slice."""
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
        if config in _config_to_task_key:
            handler_key = _config_to_task_key[config]
        else:
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


def register_task(key_or_config: str | t.Type, config_or_first_func: t.Type | t.Callable | None = None, *funcs, **kwfuncs) -> None:
    if isinstance(key_or_config, str):
        task_key = key_or_config
        config = config_or_first_func
        # config_or_first_func is the config class when key is explicit
    else:
        # Legacy: first arg is the config class
        config = key_or_config
        if config_or_first_func is not None:
            funcs = (config_or_first_func,) + funcs
        try:
            task_key = config.key
        except AttributeError:
            utils.get_logger().debug(
                f"register_task called with config lacking a 'key' ClassVar: {config}"
            )
            return

    handler_key = f"nanny_{task_key}"

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

    # Populate reverse mapping for get_task_key(config=...) lookups
    if config is not None and config not in _config_to_task_key:
        _config_to_task_key[config] = task_key

    # Register the TaskHandler (idempotent: skip if already registered)
    if handler_key not in task_registry._handlers:
        task_registry.register(handler_key, config, **task_callables)
    else:
        utils.get_logger().debug(
            f"register_task: handler '{handler_key}' already registered; skipping."
        )

    # Provide default route hook when none explicitly supplied. ``normalize`` is
    # genuinely optional (no default) — a config with nothing to normalize simply
    # omits it.
    if "route" not in hook_callables:
        hook_callables["route"] = _default_route

    # Register build hooks (idempotent: skip if already registered)
    if config not in build_hooks._registry:
        build_hooks.register(config, **hook_callables)
    else:
        utils.get_logger().debug(
            f"register_task: hooks for '{config.__name__}' already registered; skipping."
        )
