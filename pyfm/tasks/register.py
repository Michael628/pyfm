import typing as t
from pyfm.domain import task_registry, build_hooks
from pyfm.domain.task_registry import TaskHandler
from pyfm.domain.protocols import TaskHandlerProtocol
from pyfm import utils

_TASK_REGISTRY_FIELDS = frozenset(
    {"build_input_params", "create_outfile_catalog", "build_aggregator_params"}
)
_BUILD_HOOKS_FIELDS = frozenset({"preprocess", "postprocess", "validate", "preprocess_subconfig"})


def register_task(key: str, config_type: t.Type, **kwargs) -> None:
    """Register a task by splitting callables across two registries.

    Domain functions (``build_input_params``, ``create_outfile_catalog``,
    ``build_aggregator_params``) are stored in the ``task_registry``.

    Build hooks (``preprocess``, ``postprocess``, ``validate``) are stored in
    the ``build_hooks`` registry, keyed by *config_type*.

    Parameters
    ----------
    key:
        Unique string identifier for this handler (e.g. ``"hadrons_lmi"``).
    config_type:
        The config class to register.
    **kwargs:
        Any combination of the domain-function and build-hook keywords listed
        above.  Unknown keywords raise ``TypeError``.
    """
    unknown = set(kwargs) - _TASK_REGISTRY_FIELDS - _BUILD_HOOKS_FIELDS
    if unknown:
        raise TypeError(
            f"Unknown keyword(s): {unknown}. "
            f"Valid: {_TASK_REGISTRY_FIELDS | _BUILD_HOOKS_FIELDS}"
        )

    registry_kwargs = {k: v for k, v in kwargs.items() if k in _TASK_REGISTRY_FIELDS}
    hooks_kwargs = {k: v for k, v in kwargs.items() if k in _BUILD_HOOKS_FIELDS}

    task_registry.register(key, config_type, **registry_kwargs)
    if hooks_kwargs:
        build_hooks.register(config_type, **hooks_kwargs)


def get_task_handler(
    job_type: str | None = None,
    task_type: str | None = None,
    strict: bool = True,
) -> TaskHandler | None:
    """Return the ``TaskHandler`` for the given job/task type combination.

    Parameters
    ----------
    job_type:
        The top-level job type string (e.g. ``"hadrons"``).
    task_type:
        Optional sub-type string (e.g. ``"lmi"``).  When provided the lookup
        key is ``"{job_type}_{task_type}"``; otherwise it is just *job_type*.
    strict:
        When ``True`` (default) raises if the handler does not satisfy
        ``TaskHandlerProtocol``.

    Returns
    -------
    TaskHandler | None
        The registered handler, or ``None`` if not found.
    """
    if job_type is None:
        raise ValueError("Must provide job_type")

    key = "_".join([job_type, task_type]) if task_type else job_type

    try:
        handler = task_registry.get(key)
        if strict and not isinstance(handler, TaskHandlerProtocol):
            raise ValueError(
                f"Handler '{key}' does not satisfy TaskHandlerProtocol. "
                "It may be missing required build_input_params or "
                "create_outfile_catalog callables."
            )
        return handler
    except (KeyError, ValueError) as e:
        utils.get_logger().debug(str(e))
        return None


def get_task_key(
    job_type: str | None = None,
    task_type: str | None = None,
) -> str | None:
    """Return the registry key for a job/task type combination."""
    if job_type is None:
        return None
    return "_".join([job_type, task_type]) if task_type else job_type


def list_registered_types() -> t.List[str]:
    """Return all registered handler keys."""
    return task_registry.list_keys()
