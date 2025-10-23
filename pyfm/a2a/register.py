import typing as t
from pyfm.domain import HandlerRegistry, ConfigHandler

from pyfm import utils


def get_a2a_key(config: t.Type) -> str | None:
    scope = "a2a"
    try:
        handler_key = config.key
    except AttributeError:
        utils.get_logger().debug(f"Config key not provided for: {config}")
        return None

    handler_key = HandlerRegistry.get_handler_key(scope, handler_key)

    return handler_key


def get_a2a_handler(config: t.Type) -> ConfigHandler:
    handler_key = get_a2a_key(config=config)

    try:
        return HandlerRegistry.get_handler(handler_key)
    except ValueError:
        return None


def register_a2a(config: t.Type, *funcs):
    handler_key = get_a2a_key(config=config)

    HandlerRegistry.register_config(handler_key, config)
    if len(funcs) > 0:
        HandlerRegistry.register_functions(handler_key, *funcs)
