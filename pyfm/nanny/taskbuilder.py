import typing as t

from pyfm.domain import ConfigBase
from pyfm.domain.task_registry import TaskHandler
from pyfm.core.builder import build_config
from pyfm.tasks import get_task_handler, get_task_key
from pyfm.nanny.config import JobConfig, get_job_config


class Task(t.NamedTuple):
    handler: TaskHandler
    config: ConfigBase
    key: str


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
