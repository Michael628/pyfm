import typing as t
from dataclasses import dataclass

from pydantic.dataclasses import dataclass as pydantic_dataclass

from pyfm.domain import ConfigBase, SimpleConfig
from pyfm.domain.task_registry import TaskHandler
from pyfm.core.builder import build_config
from pyfm.tasks.register import get_task_handler, list_registered_types


@pydantic_dataclass(frozen=True)
class NannyConfig(SimpleConfig):
    home: str
    todo_file: str
    max_cases: int
    max_queue: int
    wait: int
    check_interval: int
    job_name_pfx: str
    scheduler: str


@pydantic_dataclass(frozen=True)
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


@dataclass(frozen=True)
class Task:
    """Immutable binding of a ``TaskHandler`` to a built config instance.

    Returned by :func:`create_task`.  Adapts the TaskHandler interface (which
    takes *config* explicitly) to the nanny callers' interface (which call
    methods without passing config).
    """

    handler: TaskHandler
    config: ConfigBase
    key: str

    def build_input_params(self) -> t.Any:
        return self.handler.build_input_params(self.config)

    def create_outfile_catalog(self) -> t.Any:
        return self.handler.create_outfile_catalog(self.config)

    def build_aggregator_params(self, *args, **kwargs) -> t.Any:
        return self.handler.build_aggregator_params(self.config, *args, **kwargs)

    def format_string(self, to_format: str) -> str:
        return self.config.format_string(to_format)


def get_nanny_config(yaml_params: t.Dict[str, t.Any]) -> NannyConfig:
    nanny_params = yaml_params.get("shared_params", {})
    nanny_params |= yaml_params["nanny"]
    nanny_params |= yaml_params["submit"]
    nanny_params |= yaml_params.get("files", {})
    return build_config(NannyConfig, nanny_params)


def get_job_config(job_step: str, yaml_params: t.Dict[str, t.Any]) -> JobConfig:
    job_defaults = yaml_params.get("shared_params", {})
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
    """Build a config and bind it to its handler, returning a :class:`Task`.

    The returned ``Task`` provides convenience methods that call handler
    functions with the config passed explicitly, while keeping the nanny
    caller interface unchanged.
    """
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

    job_type = job_config.job_type
    task_type = job_config.task_type

    handler = get_task_handler(job_type, task_type)
    assert handler is not None, f"No handler found for {job_type}, {task_type}"

    config_type = handler.config_type
    file_params = yaml_params.get("files", {})

    # Merge task_params into global_params under '_tasks' key
    config_params = global_params | {"_tasks": task_params}

    config = build_config(config_type, config_params, file_params)

    key = "_".join([job_type, task_type]) if task_type else job_type
    return Task(handler=handler, config=config, key=key)
