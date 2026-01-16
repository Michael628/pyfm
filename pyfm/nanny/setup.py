import typing as t

from pyfm.domain import (
    ConfigBase,
    ConfigHandler,
    ConfigPostprocessorProtocol,
)
from pyfm.core.builder import build_config
from pyfm.tasks import get_task_handler, register_task


# TODO: Consolidate layout, and job params into a config object
def get_layout_params(
    job_step: str, yaml_params: t.Dict[str, t.Any]
) -> t.Dict[str, t.Any]:
    if "submit" not in yaml_params or "layout" not in yaml_params["submit"]:
        raise ValueError("No `submit` parameters provided.")

    layout = yaml_params.get("submit").get("layout")
    if job_step not in layout:
        raise ValueError(f"No layout parameters provided for `{job_step}`.")
    return layout | layout[job_step]


def get_job_params(
    job_step: str, yaml_params: t.Dict[str, t.Any]
) -> t.Dict[str, t.Any]:
    job_defaults = {
        "job_type": "hadrons",
        "task_type": "lmi",
    }
    if "job_setup" not in yaml_params:
        raise ValueError("No `job_setup` parameters provided.")
    if job_step not in yaml_params["job_setup"]:
        raise ValueError(f"No `job_setup` parameters provided for `{job_step}`.")

    job_params = job_defaults | yaml_params.get("job_setup").get(job_step)

    job_type, task_type = job_params["job_type"], job_params["task_type"]

    if job_type != "hadrons":
        job_params["task_type"] = task_type = None

    if get_task_handler(job_type, task_type) is None:
        raise ValueError(f"No task handler found for {job_type}, {task_type}")

    return job_params


def get_task_params(
    job_step: str, yaml_params: t.Dict[str, t.Any], defaults: t.Dict[str, t.Any] | None
) -> t.Tuple[t.Dict[str, t.Any], t.Dict[str, t.Any]]:
    """
    Returns:
        (global_params, task_configs)
        - global_params: Flattened parameter hierarchy for overrides
        - task_configs: Unflattened task configuration structure
    """
    if defaults is None:
        defaults = {}

    job_params = get_job_params(job_step, yaml_params)

    job_type = job_params["job_type"]

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
        job_params.get("params", {})
    )

    # Keep task configs separate
    task_configs = job_params.get("tasks", {})

    return global_params, task_configs


def create_task(
    job_step: str,
    yaml_params: t.Dict[str, t.Any],
    series: str | None = None,
    cfg: str | None = None,
) -> ConfigHandler:
    """Create a new ConfigHandler. If the relevant task type is found, the returned object will have
    methods corresponding to all functions assigned to the task in the corresponding task file.
    """
    param_defaults = {
        "logging_level": "INFO",
    }
    if series:
        param_defaults["series"] = series
    if cfg:
        param_defaults["cfg"] = cfg

    # Get separated params
    global_params, task_configs = get_task_params(
        job_step, yaml_params, defaults=param_defaults
    )

    job_type, task_type = map(
        get_job_params(job_step, yaml_params).get, ["job_type", "task_type"]
    )

    handler = get_task_handler(job_type, task_type)
    assert handler is not None, f"No get_task_handler found for {job_type}, {task_type}"

    config_type = handler.get_config_type()

    file_params = yaml_params.get("files", {})

    # Merge task_configs into global_params under '_tasks' key
    config_params = global_params | {"_tasks": task_configs}

    handler.config = build_config(
        config_type,
        config_params,
        file_params,
        get_handler=lambda x: get_task_handler(config=x),
    )

    if isinstance(handler, ConfigPostprocessorProtocol):
        handler.config = handler.postprocess_config()

    # Register a default function for formatting variables found in strings in config parameters
    def format_string(config: ConfigBase, to_format: str) -> str:
        try:
            return config.format_string(to_format)
        except KeyError as e:
            raise ValueError(f"Couldn't find key in parameters: {e}")

    register_task(handler.get_config_type(), format_string)

    return handler
