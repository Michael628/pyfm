import typing as t
from pydantic.dataclasses import dataclass

from pyfm.domain import ConfigBase, SimpleConfig
from pyfm.domain.task_registry import TaskHandler
from pyfm.core.builder import build_config
from pyfm.tasks import get_task_handler, get_task_key, list_registered_types


@dataclass(frozen=True)
class NannyConfig(SimpleConfig):
    home: str
    todo_file: str
    max_cases: int
    max_queue: int
    wait: int
    check_interval: int
    job_name_pfx: str
    scheduler: str


@dataclass(frozen=True)
class JobConfig(SimpleConfig):
    run: str
    job_type: str
    tasks: t.Dict[str, t.Any]
    step: str
    io: str
    wall_time: str
    ppn: int
    nodes: int
    lattice: t.List[int]
    geom: t.List[int]
    params: t.Dict[str, t.Any]
    node_minimum: int | None = None
    task_type: str | None = None
    barrier: bool = True


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


def get_job_config(job_step: str, yaml_params: t.Dict[str, t.Any]) -> JobConfig:
    job_defaults = yaml_params.get("shared_params", {})
    # job_defaults |= {"job_type": "hadrons", "task_type": "lmi", "step": job_step}
    job_defaults |= {"step": job_step, "params": {}}
    if "job_setup" not in yaml_params:
        raise ValueError("No `job_setup` parameters provided.")
    if job_step not in yaml_params["job_setup"]:
        raise ValueError(f"No `job_setup` parameters provided for `{job_step}`.")

    job_params = job_defaults | yaml_params.get("job_setup").get(job_step)

    job_type, task_type = job_params.get("job_type", None), job_params.get(
        "task_type", None
    )

    if get_task_handler(job_type, task_type) is None:
        raise ValueError(
            f"No task handler found for job_type={job_type!r}, task_type={task_type!r}. "
            f"Registered types: {list_registered_types()}"
        )

    job_params |= yaml_params["submit"]["layout"]
    job_params |= yaml_params["submit"]["layout"].get(job_step, {})
    return build_config(JobConfig, job_params)


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
    task_params = job_config.tasks

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
