import typing as t

from pyfm.domain import (
    ConfigBase,
    ConfigHandler,
    ConfigPostprocessorProtocol,
)
from pyfm.builder import build_config
from pyfm.tasks import get_task_handler, register_task


def get_task_params(
    job_step: str, yaml_params: t.Dict[str, t.Any], defaults: t.Dict[str, t.Any] | None
) -> t.Dict[str, t.Any]:
    if defaults is None:
        defaults = {}

    job_params = yaml_params.get("job_setup", {}).get(job_step, {})
    job_type = job_params.get("job_type", "")

    task_params = (
        defaults
        |
        # Load common submit parameters
        yaml_params.get("submit_params", {})
        |
        # Load job-type parameters
        yaml_params.get(f"{job_type}_params", {})
        |
        # Load job-specific overrides
        job_params.get("params", {})
        |
        # Load task-specific parameters
        job_params.get("tasks", {})
    )
    return task_params


def create_task(
    job_step: str,
    yaml_params: t.Dict[str, t.Any],
    series: str | None = None,
    cfg: str | None = None,
) -> ConfigHandler:
    """Create a new ConfigHandler. If the relevant task type is found, the returned object will have
    methods corresponding to all functions assigned to the task in the corresponding task file.
    """
    job_type = yaml_params.get("job_setup", {}).get(job_step, {}).get("job_type", "")
    task_type = yaml_params.get("job_setup", {}).get(job_step, {}).get("task_type", "")

    handler = get_task_handler(job_type, task_type)

    config_type = handler.get_config_type()

    param_defaults = {"logging_level": "INFO"}
    if series:
        param_defaults["series"] = series
    if cfg:
        param_defaults["cfg"] = cfg

    task_params = get_task_params(job_step, yaml_params, defaults=param_defaults)

    file_params = yaml_params.get("files", {})
    handler.config = build_config(
        config_type,
        task_params,
        file_params,
        get_handler=lambda x: get_task_handler(config=x),
    )

    if isinstance(handler, ConfigPostprocessorProtocol):
        handler.config = handler.postprocess_config()

    def format_string(config: ConfigBase, to_format: str) -> str:
        return config.format_string(to_format)

    register_task(handler.config, format_string)

    return handler
