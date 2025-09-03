import typing as t
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable, Union, List, Optional, Dict
import pandas as pd

from pyfm import utils, setup_logging
from pyfm.nanny import tasks
from pyfm.nanny import SubmitConfig


@runtime_checkable
class TaskConfigProtocol(Protocol):
    def input_params(
        self, submit_config: SubmitConfig
    ) -> Union[t.Tuple[List[Dict], Optional[List[str]]], Dict[str, Dict]]: ...

    def processing_params(self, submit_config: SubmitConfig) -> Dict: ...

    def catalog_files(self, submit_config: SubmitConfig) -> pd.DataFrame: ...

    def bad_files(self, submit_config: SubmitConfig) -> List[str]: ...


# ============Job Configuration===========
@dataclass
class JobConfig:
    """Holds job configuration information.

    Attributes
    ----------
    tasks : TaskConfigProtocol
        Task configuration implementing the TaskConfigProtocol interface
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

    tasks: TaskConfigProtocol
    io: str
    job_type: str
    task_type: t.Optional[str] = None
    params: t.Dict = field(default_factory=dict)

    def input_params(
        self, submit_config: SubmitConfig
    ) -> Union[t.Tuple[List[Dict], Optional[List[str]]], Dict[str, Dict]]:
        return self.tasks.input_params(submit_config)

    def processing_params(self, submit_config: SubmitConfig) -> Dict:
        return self.tasks.processing_params(submit_config)

    def catalog_files(self, submit_config: SubmitConfig) -> pd.DataFrame:
        return self.tasks.catalog_files(submit_config)

    def bad_files(self, submit_config: SubmitConfig) -> List[str]:
        return self.tasks.bad_files(submit_config)

    def has_good_output(self, submit_config: SubmitConfig) -> bool:
        """Check if all expected output files are present and valid.

        Parameters
        ----------
        submit_config : SubmitConfig
            Configuration parameters for submitted job.

        Returns
        -------
        bool
            True if all files are present and valid, False otherwise.
        """
        return len(self.bad_files(submit_config)) == 0

    def get_file_summary(self, submit_config: SubmitConfig) -> Dict[str, int]:
        """Get a summary of file status for this job configuration.

        Parameters
        ----------
        submit_config : SubmitConfig
            Configuration parameters for submitted job.

        Returns
        -------
        Dict[str, int]
            Dictionary with counts: {'total': N, 'existing': N, 'good': N, 'bad': N}
        """
        df = self.catalog_files(submit_config)
        bad_file_paths = set(self.bad_files(submit_config))

        total = len(df)
        existing = int(df["exists"].sum()) if "exists" in df.columns else 0
        bad = len(bad_file_paths)
        good = existing - bad

        return {"total": total, "existing": existing, "good": good, "bad": bad}

    @classmethod
    def from_dict(cls, kwargs) -> "JobConfig":
        """Creates a new instance of JobConfig from a dictionary.

        Raises
        ------
        TypeError
            If the created task configuration doesn't implement TaskConfigProtocol.
        """

        params = utils.deep_copy_dict(kwargs)

        params.pop("run")
        params.pop("wall_time")

        if "job_type" not in params:
            params["job_type"] = "hadrons"
        if params["job_type"] == "hadrons":
            if "task_type" not in params:
                params["task_type"] = "lmi"

        task_type = params.get("task_type", None)

        task_params = params.get("tasks", {})
        task_builder: t.Callable = tasks.get_task_factory(params["job_type"], task_type)
        task_config = task_builder(task_params)

        params["tasks"] = task_config
        instance = cls(**params)

        # Additional validation
        instance.validate_task_config()

        return instance

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


# ============Legacy functions===========
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

    setup_logging(submit_params.get("logging_level", "INFO"))

    return tasks.get_submit_factory(job_config.job_type)(**submit_params, **kwargs)
