from dataclasses import dataclass
from typing import Callable, Dict, Optional, Type


@dataclass(frozen=True)
class BuildHooks:
    """Frozen dataclass holding optional build lifecycle hooks for a config type."""

    preprocess: Optional[Callable] = None
    postprocess: Optional[Callable] = None
    validate: Optional[Callable] = None
    preprocess_subconfig: Optional[Callable] = None


# ---------------------------------------------------------------------------
# Module-level singleton registry
# ---------------------------------------------------------------------------

_registry: Dict[Type, BuildHooks] = {}

_VALID_HOOKS = {"preprocess", "postprocess", "validate", "preprocess_subconfig"}


def register(config_type: Type, **hooks: Callable) -> None:
    """Register build hooks for *config_type*.

    Parameters
    ----------
    config_type:
        The config class to attach hooks to.
    **hooks:
        Keyword arguments accepted: ``preprocess``, ``postprocess``,
        ``validate``, ``preprocess_subconfig``.  Any other key raises a
        ``TypeError``.

    Raises
    ------
    TypeError
        If an unknown hook keyword is provided.
    ValueError
        If *config_type* already has hooks registered.
    """
    unknown = set(hooks) - _VALID_HOOKS
    if unknown:
        raise TypeError(
            f"Unknown hook keyword(s): {unknown}. "
            f"Valid keywords are: {_VALID_HOOKS}"
        )

    if config_type in _registry:
        raise ValueError(
            f"Hooks already registered for config type '{config_type.__name__}'. "
            "Use clear() to reset the registry in tests."
        )

    _registry[config_type] = BuildHooks(**hooks)


def get(config_type: Type) -> Optional[BuildHooks]:
    """Return the BuildHooks for *config_type*, or None if not registered."""
    return _registry.get(config_type)


def clear() -> None:
    """Remove all registered hooks. Intended for test isolation."""
    _registry.clear()
