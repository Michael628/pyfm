from dataclasses import dataclass
from typing import Callable, ClassVar, Dict, FrozenSet, Optional, Type

from pyfm.domain.conftypes import ConfigBase


@dataclass(frozen=True)
class TaskHandler:
    """Frozen dataclass mapping a config type to its optional build callables.

    Each callable receives the config instance as its first explicit parameter.
    Fields that are ``None`` are treated as absent for the purposes of
    ``isinstance`` / ``@runtime_checkable`` Protocol checks: accessing them
    raises ``AttributeError`` so that structural-subtype checks work correctly.

    Attributes
    ----------
    config_type:
        The ``ConfigBase`` subclass this handler is associated with.
    build_input_params:
        Callable ``(config, ...) -> Any`` that produces external-program input.
    create_outfile_catalog:
        Callable ``(config, ...) -> Any`` that enumerates expected output files.
    build_aggregator_params:
        Callable ``(config, ...) -> Any`` that provides aggregation parameters.
    """

    config_type: Type[ConfigBase]
    build_input_params: Optional[Callable] = None
    create_outfile_catalog: Optional[Callable] = None
    build_aggregator_params: Optional[Callable] = None

    # Names of the optional callable fields.  Stored as a ClassVar frozenset so
    # it doesn't participate in __init__, __eq__, __repr__, or __hash__.
    _CALLABLE_FIELDS: ClassVar[FrozenSet[str]] = frozenset(
        {"build_input_params", "create_outfile_catalog", "build_aggregator_params"}
    )

    def __getattribute__(self, name: str):
        val = object.__getattribute__(self, name)
        # Make None-valued callable fields behave as absent attributes so that
        # runtime_checkable Protocol isinstance checks work correctly.
        if name in TaskHandler._CALLABLE_FIELDS and val is None:
            raise AttributeError(
                f"TaskHandler has no callable '{name}' — it was not registered."
            )
        return val


# ---------------------------------------------------------------------------
# Module-level singleton registry
# ---------------------------------------------------------------------------

_handlers: Dict[str, TaskHandler] = {}

_VALID_CALLABLES = {"build_input_params", "create_outfile_catalog", "build_aggregator_params"}


def register(
    key: str,
    config_type: Type[ConfigBase],
    **callables: Callable,
) -> None:
    """Register a new ``TaskHandler`` under *key*.

    Parameters
    ----------
    key:
        Unique string identifier for this handler.
    config_type:
        The ``ConfigBase`` subclass the handler is associated with.
    **callables:
        Optional callables to attach.  Accepted keywords:
        ``build_input_params``, ``create_outfile_catalog``,
        ``build_aggregator_params``.  Any other keyword raises ``TypeError``.

    Raises
    ------
    TypeError
        If an unrecognised callable keyword is supplied.
    ValueError
        If a handler is already registered under *key*.
    """
    unknown = set(callables) - _VALID_CALLABLES
    if unknown:
        raise TypeError(
            f"Unknown callable keyword(s): {unknown}. Valid: {_VALID_CALLABLES}"
        )
    if key in _handlers:
        raise ValueError(
            f"Handler already registered for key '{key}'. "
            "Use clear() to reset the registry in tests."
        )
    _handlers[key] = TaskHandler(config_type=config_type, **callables)


def get(key: str) -> TaskHandler:
    """Return the ``TaskHandler`` registered under *key*.

    Raises
    ------
    KeyError
        If no handler has been registered under *key*.
    """
    if key not in _handlers:
        raise KeyError(
            f"No handler registered for key '{key}'. "
            f"Available keys: {list(_handlers)}"
        )
    return _handlers[key]


def list_keys() -> list:
    """Return all registered handler keys."""
    return list(_handlers)


def clear() -> None:
    """Remove all registered handlers. Intended for test isolation."""
    _handlers.clear()
