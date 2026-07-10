import typing as t
from enum import auto
from pydantic.dataclasses import dataclass

from pyfm.domain import ConfigBase, SimpleConfig, SerializableEnum, build_hooks
from pyfm.domain.task_registry import TaskHandler
from pyfm.core.builder import build_config
from pyfm.tasks import get_task_handler, get_task_key
from pyfm.nanny.jobconfig import (
    JobConfig,
    JobBundle,
    get_job_config,
    build_job_configs,
)
from pyfm import utils


class Scheduler(SerializableEnum):
    LSF = auto()
    PBS = auto()
    SLURM = auto()
    INTERACTIVE = auto()
    COBALT = auto()


@dataclass(frozen=True)
class NannyConfig(SimpleConfig):
    home: str
    todo_file: str
    max_queue: int
    wait: int
    check_interval: int
    job_name_pfx: str
    scheduler: Scheduler


def warn_moved_max_cases(params: t.Dict) -> t.Dict:
    """Normalize hook for NannyConfig: warn when the legacy ``nanny.max_cases``
    key is present.

    Must be a ``normalize`` hook (runs on raw params before construction), not
    ``validate`` — pydantic drops unknown keys at construction, so a removed
    ``max_cases`` field is invisible to ``validate``. The legacy value is ignored;
    the user must set ``submit:->resources:->max_cases`` or a per-step override.
    """
    if "max_cases" in params:
        utils.get_logger().warning(
            "`max_cases` has moved from the `nanny:` stanza to the "
            "`submit:->resources:` stanza (global) or a per-step override "
            "(`submit:->resources:-><step>:->max_cases`). The `nanny.max_cases` "
            "value is no longer read."
        )
    return params


build_hooks.register(NannyConfig, normalize=warn_moved_max_cases)


class Task(t.NamedTuple):
    handler: TaskHandler
    config: ConfigBase
    key: str


def get_nanny_config(yaml_params: t.Dict[str, t.Any]) -> NannyConfig:
    nanny_params = yaml_params.get("shared_params", {})
    nanny_params |= yaml_params["nanny"]
    nanny_params |= yaml_params["submit"]
    nanny_params |= yaml_params.get("files", {})
    return build_config(NannyConfig, nanny_params)


def get_task_params(
    job_config: JobConfig,
    yaml_params: t.Dict[str, t.Any],
    defaults: t.Dict[str, t.Any] | None,
) -> t.Tuple[t.Dict[str, t.Any], t.Dict[str, t.Any]]:
    """
    Returns:
        (global_params, task_params)
        - global_params: Flattened parameter hierarchy for overrides
        - task_params: Unflattened task configuration structure
    """
    if defaults is None:
        defaults = {}

    job_type = job_config.job_type

    # Build flattened global params (WITHOUT tasks)
    global_params = (
        defaults
        |
        # Load common shared parameters (legacy)
        yaml_params.get("submit_params", {})
        |
        # Load common shared parameters
        yaml_params.get("shared_params", {})
        |
        # Load job-type parameters
        yaml_params.get(f"{job_type}_params", {})
        |
        # Load job-specific overrides
        job_config.params
    )

    # Keep task configs separate
    task_params = job_config.tasks if job_config.tasks is not None else {}

    return global_params, task_params


def create_task(
    job_step: str,
    yaml_params: t.Dict[str, t.Any],
    series: str | None = None,
    cfg: str | None = None,
) -> Task:
    """Build a task config and return a Task NamedTuple."""
    param_defaults = {
        "logging_level": "INFO",
    }
    if series:
        param_defaults["series"] = series
    if cfg:
        param_defaults["cfg"] = cfg

    job_config = get_job_config(job_step, yaml_params)
    global_params, task_params = get_task_params(
        job_config, yaml_params, defaults=param_defaults
    )

    job_type, task_type = map(
        lambda x: getattr(job_config, x), ["job_type", "task_type"]
    )

    handler = get_task_handler(job_type, task_type)
    assert handler is not None, f"No get_task_handler found for {job_type}, {task_type}"

    file_params = yaml_params.get("files", {})
    config_params = global_params | {"_preprocessor": task_params}

    config = build_config(
        handler.config_type,
        config_params,
        file_params,
    )

    task_key = get_task_key(job_type, task_type)
    return Task(handler=handler, config=config, key=task_key)
