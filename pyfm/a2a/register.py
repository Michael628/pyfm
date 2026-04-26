import typing as t

from pyfm.domain import task_registry
from pyfm.domain.task_registry import TaskHandler
from pyfm import utils


def get_a2a_key(config: t.Type) -> str | None:
    scope = "a2a"
    try:
        handler_key = config.key
    except AttributeError:
        utils.get_logger().debug(f"Config key not provided for: {config}")
        return None

    return f"{scope}_{handler_key}"


def get_a2a_handler(config: t.Type) -> TaskHandler | None:
    handler_key = get_a2a_key(config=config)
    if handler_key is None:
        return None
    try:
        return task_registry.get(handler_key)
    except KeyError:
        return None


def register_a2a(config: t.Type, *funcs) -> None:
    handler_key = get_a2a_key(config=config)
    if handler_key is None:
        return

    if handler_key in task_registry._handlers:
        utils.get_logger().debug(
            f"register_a2a: handler '{handler_key}' already registered; skipping."
        )
        return

    _TASK_CALLABLE_NAMES = frozenset(
        {"build_input_params", "create_outfile_catalog", "build_aggregator_params"}
    )
    callables = {}
    for fn in funcs:
        name = getattr(fn, "__name__", None)
        if name in _TASK_CALLABLE_NAMES:
            callables[name] = fn
        else:
            utils.get_logger().debug(
                f"register_a2a: ignoring function '{name}' — not a recognised callable name."
            )

    task_registry.register(handler_key, config, **callables)
