from dataclasses import dataclass
from typing import Callable, Dict, Optional, Type


@dataclass(frozen=True)
class BuildHooks:
    """Frozen dataclass holding optional build lifecycle hooks for a config type."""

    preprocess: Optional[Callable] = None
    postprocess: Optional[Callable] = None
    validate: Optional[Callable] = None


class BuildHooksRegistry:
    """Global singleton registry mapping config_type -> BuildHooks.

    Enforces a 1:1 mapping — registering hooks for a config type that already
    has an entry raises a ValueError at import time.  Call clear() in tests to
    reset state between test runs.
    """

    _registry: Dict[Type, BuildHooks] = {}

    @classmethod
    def register(cls, config_type: Type, **hooks: Callable) -> None:
        """Register build hooks for *config_type*.

        Parameters
        ----------
        config_type:
            The config class to attach hooks to.
        **hooks:
            Keyword arguments accepted: ``preprocess``, ``postprocess``,
            ``validate``.  Any other key raises a ``TypeError``.

        Raises
        ------
        TypeError
            If an unknown hook keyword is provided.
        ValueError
            If *config_type* already has hooks registered.
        """
        valid_fields = {"preprocess", "postprocess", "validate"}
        unknown = set(hooks) - valid_fields
        if unknown:
            raise TypeError(
                f"Unknown hook keyword(s): {unknown}. "
                f"Valid keywords are: {valid_fields}"
            )

        if config_type in cls._registry:
            raise ValueError(
                f"Hooks already registered for config type '{config_type.__name__}'. "
                "Use clear() to reset the registry in tests."
            )

        cls._registry[config_type] = BuildHooks(**hooks)

    @classmethod
    def get(cls, config_type: Type) -> Optional[BuildHooks]:
        """Return the BuildHooks for *config_type*, or None if not registered."""
        return cls._registry.get(config_type)

    @classmethod
    def clear(cls) -> None:
        """Remove all registered hooks. Intended for test isolation."""
        cls._registry.clear()
