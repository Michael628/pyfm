import typing as t
from functools import partial
from pydantic import Field

from pyfm.domain import (
    ConfigBuilder,
    ConfigBase,
    CompositeConfig,
    TaskRegistry,
    TaskHandler,
    ConfigProcessorProtocol,
)


def get_task_key(job_step: str, yaml_data: t.Dict[str, t.Any]) -> str:
    job_type = yaml_data.get("job_setup", {}).get(job_step, {}).get("job_type", "")
    task_type = yaml_data.get("job_setup", {}).get(job_step, {}).get("task_type", "")
    return TaskRegistry.get_task_key(job_type, task_type)


def get_handler(job_step: str, yaml_data: t.Dict[str, t.Any]) -> TaskHandler:
    task_key = get_task_key(job_step, yaml_data)
    return TaskRegistry.get_handler(task_key)


def new_task_builder(
    config_type,
    job_step: str,
    yaml_data: t.Dict[str, t.Any],
    task_preproc: t.List[t.Callable] | None = None,
) -> ConfigBuilder:
    """Setup a new task builder initialized with standerd yaml parameters.

    If `task_mapping` is provided, the task data under the key `task_mapping.key` will be remapped
    to `task_mapping.remap` in the resulting config.
    """

    if task_preproc is None:
        task_preproc = []

    config_builder = ConfigBuilder(config_type)

    job_data = yaml_data.get("job_setup", {}).get(job_step, {})
    job_type = job_data.get("job_type", "")

    task_data = job_data.get("tasks", {})

    for preproc in task_preproc:
        task_data = preproc(task_data)

    return (
        config_builder.with_field("logging_level", "INFO")
        # Load common submit parameters
        .with_yaml_section(yaml_data, "submit_params")
        # Load job-type parameters
        .with_yaml_section(yaml_data, f"{job_type}_params")
        # Load task-specific parameters
        .with_yaml(task_data)
        # Load job-specific overrides
        .with_yaml_section(job_data, "params")
    )


def create_task(
    job_step: str,
    yaml_data: t.Dict[str, t.Any],
    series: str | None = None,
    cfg: str | None = None,
) -> TaskHandler:
    """Create a new TaskHandler. If the relevant task type is found, the returned object will have
    methods corresponding to all functions assigned to the task in the corresponding task file.
    """

    def build_config(
        config_type, task_preproc: t.List[t.Callable] | None = None
    ) -> ConfigBase:
        """Build config to assign to the task handler. Any 'config' argument in the functions
        assigned to the task will have this config object injected into it.
        """

        if task_preproc is None:
            task_preproc = []

        handler = get_handler(job_step, yaml_data)

        if issubclass(config_type, CompositeConfig):
            subconfigs = {}
            for k, v in config_type.get_subconfigs().items():
                if isinstance(handler, ConfigProcessorProtocol):
                    preproc = task_preproc + [
                        partial(handler.preprocess_params, subconfig=k)
                    ]
                subconfigs[k] = build_config(v, task_preproc=preproc)

            return (
                new_task_builder(config_type, job_step, yaml_data)
                .with_yaml(subconfigs)
                .build()
            )

        preproc = task_preproc
        if isinstance(handler, ConfigProcessorProtocol):
            preproc = task_preproc + [handler.preprocess_params]

        files = yaml_data.get("files", {})
        return (
            new_task_builder(config_type, job_step, yaml_data, preproc)
            .with_field("series", series)
            .with_field("cfg", cfg)
            .with_files(files)
            .build()
        )

    handler = get_handler(job_step, yaml_data)

    config_type = handler.get_config_type()

    handler.config = build_config(config_type)

    if isinstance(handler, ConfigProcessorProtocol):
        handler.config = handler.postprocess_config()

    def format_string(config: ConfigBase, to_format: str) -> str:
        return config.format_string(to_format)

    task_key = get_task_key(job_step, yaml_data)
    TaskRegistry.register_function(task_key, format_string)

    return handler
