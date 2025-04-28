import typing as t
from dataclasses import field
from pydantic.dataclasses import dataclass

from pyfm import utils
from pyfm.nanny import tasks
from pyfm.nanny import SubmitConfig, TaskBase


# ============Job Configuration===========
@dataclass
class JobConfig:
    """Holds job configuration information.

    Attributes
    ----------
    tasks : TaskBase
        Parameters for task `task_type`
    io : string, optional
       Used in input/output file names
    job_type : string
        Type of job to run. Must correspond to file/directory
        in nanny/tasks/
    task_type : string
        `tasks` argument must be the name of an existing file, <tasks>.py,
        in subdirectory nanny/tasks/<job_type>/ or nanny/tasks/ if job_type is None.
    params : dict, optional
        Overrides values provided to SubmitConfig constructor when called through
        get_submit_config.

    """

    tasks: TaskBase
    io: str
    job_type: str
    task_type: t.Optional[str] = None
    params: t.Dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, kwargs) -> "JobConfig":
        """Creates a new instance of JobConfig from a dictionary."""

        params = utils.deep_copy_dict(kwargs)

        if "job_type" not in params:
            params["job_type"] = "hadrons"
        if params["job_type"] == "hadrons":
            if "task_type" not in params:
                params["task_type"] = "lmi"

        task_type = params.get("task_type", None)

        task_params = params.get("tasks", {})
        task_builder: t.Callable = tasks.get_task_factory(params["job_type"], task_type)
        params["tasks"] = task_builder(task_params)

        return cls(**params)

    def get_infile(self, submit_config: SubmitConfig) -> str:
        """Returns input file including extension. Formats string based on
        varirables provided by `submit_config.`

        Parameters
        ----------
        submit_config : SubmitConfig
           Configuration parameters for submitted job.

        Returns
        -------
        str
          Formatted input file string.

        """
        ext = {
            "smear": "{series}.{cfg}.txt",
            "hadrons": "{series}.{cfg}.xml",
            "contract": "{series}.{cfg}.yaml",
        }
        return f"{self.io}-{ext[self.job_type]}".format(**submit_config.string_dict())


# ============Convenience functions===========
def get_job_config(param: t.Dict, step: str) -> JobConfig:
    try:
        return JobConfig.from_dict(param["job_setup"][step])
    except KeyError:
        raise NotImplementedError(f"Job step `{step}` missing from param file.")


def get_submit_config(param: t.Dict, job_config: JobConfig, **kwargs) -> SubmitConfig:
    submit_params = utils.deep_copy_dict(param["submit_params"])
    additional_params = job_config.job_type + "_params"

    if additional_params in param:
        submit_params.update(param[additional_params])
    if job_config.params:
        submit_params.update(job_config.params)

    assert "files" not in submit_params
    submit_params["files"] = param["files"]

    return tasks.get_submit_factory(job_config.job_type)(**submit_params, **kwargs)


def input_params(
    job_config: JobConfig, *args, **kwargs
) -> t.Tuple[t.List[t.Dict], t.Optional[t.List[str]]]:
    return tasks.input_params(
        job_config.job_type, job_config.task_type, job_config.tasks, *args, **kwargs
    )


def processing_params(
    job_config: JobConfig, *args, **kwargs
) -> t.Tuple[t.List[t.Dict], t.Optional[t.List[str]]]:
    return tasks.processing_params(
        job_config.job_type, job_config.task_type, job_config.tasks, *args, **kwargs
    )


def bad_files(job_config: JobConfig, *args, **kwargs) -> t.List[str]:
    return tasks.bad_files(
        job_config.job_type, job_config.task_type, job_config.tasks, *args, **kwargs
    )


if __name__ == "__main__":
    lmi_builder = tasks.get_task_factory("hadrons", "lmi")

    a = lmi_builder.from_dict(
        {"epack": {"load": False}, "meson": {"gamma": "onelink", "mass": "l"}}
    )
