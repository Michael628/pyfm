import typing as t
import warnings
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable, Union, List, Optional, Dict, Any
import pandas as pd

from pyfm import utils, setup_logging
from pyfm.nanny import tasks
from pyfm.nanny import SubmitConfig


@runtime_checkable
class TaskConfigProtocol(Protocol):
    """Protocol defining the common interface for task configurations.

    This protocol ensures that all task configurations provide consistent
    methods for generating input parameters, processing configurations,
    file catalogs, and identifying problematic files.
    """

    def input_params(
        self, submit_config: SubmitConfig
    ) -> Union[t.Tuple[List[Dict], Optional[List[str]]], Dict[str, Dict]]:
        """Generate input parameters for job execution.

        Parameters
        ----------
        submit_config : SubmitConfig
            Configuration parameters for submitted job.

        Returns
        -------
        Union[Tuple[List[Dict], Optional[List[str]]], Dict[str, Dict]]
            Either a tuple of (modules, schedule) for LMI tasks or
            a dictionary of module configurations for other tasks.
        """
        ...

    def processing_params(self, submit_config: SubmitConfig) -> Dict:
        """Generate processing parameters for data analysis.

        Parameters
        ----------
        submit_config : SubmitConfig
            Configuration parameters for submitted job.

        Returns
        -------
        Dict
            Processing configuration with 'run' key containing
            list of processing tasks.
        """
        ...

    def catalog_files(self, submit_config: SubmitConfig) -> pd.DataFrame:
        """Generate file catalog with metadata for tracking outputs.

        Parameters
        ----------
        submit_config : SubmitConfig
            Configuration parameters for submitted job.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: filepath, exists, file_size, good_size
        """
        ...

    def bad_files(self, submit_config: SubmitConfig) -> List[str]:
        """Identify incomplete or corrupted files that need regeneration.

        Parameters
        ----------
        submit_config : SubmitConfig
            Configuration parameters for submitted job.

        Returns
        -------
        List[str]
            List of file paths that are incomplete or corrupted.
        """
        ...


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
        """Generate input parameters for job execution.

        Parameters
        ----------
        submit_config : SubmitConfig
            Configuration parameters for submitted job.

        Returns
        -------
        Union[Tuple[List[Dict], Optional[List[str]]], Dict[str, Dict]]
            Input parameters for job execution. LMI tasks return (modules, schedule),
            other tasks return module configurations.
        """
        return self.tasks.input_params(submit_config)

    def processing_params(self, submit_config: SubmitConfig) -> Dict:
        """Generate processing parameters for data analysis.

        Parameters
        ----------
        submit_config : SubmitConfig
            Configuration parameters for submitted job.

        Returns
        -------
        Dict
            Processing configuration with 'run' key containing processing tasks.
        """
        return self.tasks.processing_params(submit_config)

    def catalog_files(self, submit_config: SubmitConfig) -> pd.DataFrame:
        """Generate file catalog with metadata for tracking outputs.

        Parameters
        ----------
        submit_config : SubmitConfig
            Configuration parameters for submitted job.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: filepath, exists, file_size, good_size
        """
        return self.tasks.catalog_files(submit_config)

    def bad_files(self, submit_config: SubmitConfig) -> List[str]:
        """Identify incomplete or corrupted files that need regeneration.

        Parameters
        ----------
        submit_config : SubmitConfig
            Configuration parameters for submitted job.

        Returns
        -------
        List[str]
            List of file paths that are incomplete or corrupted.
        """
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
        existing = int(df['exists'].sum()) if 'exists' in df.columns else 0
        bad = len(bad_file_paths)
        good = existing - bad
        
        return {
            'total': total,
            'existing': existing, 
            'good': good,
            'bad': bad
        }

    def validate_task_config(self) -> None:
        """Validate that the task configuration implements the required protocol.

        Raises
        ------
        TypeError
            If the task configuration doesn't implement TaskConfigProtocol.
        ValueError
            If the task configuration is invalid for the job type.
        """
        # Since tasks is now typed as TaskConfigProtocol, this should always pass
        # but we keep it for runtime safety
        if not isinstance(self.tasks, TaskConfigProtocol):
            raise TypeError(
                f"Task config {type(self.tasks)} does not implement TaskConfigProtocol. "
                f"Expected protocol methods: input_params, processing_params, "
                f"catalog_files, bad_files"
            )

        # Validate that all required methods are callable
        required_methods = ['input_params', 'processing_params', 'catalog_files', 'bad_files']
        for method_name in required_methods:
            if not hasattr(self.tasks, method_name):
                raise TypeError(f"Task config {type(self.tasks)} missing required method: {method_name}")
            if not callable(getattr(self.tasks, method_name)):
                raise TypeError(f"Task config {type(self.tasks)} method {method_name} is not callable")

        # Additional validation based on job_type
        if self.job_type == "hadrons" and not hasattr(self.tasks, "mass"):
            raise ValueError("Hadrons tasks must have a 'mass' property")

    @classmethod
    def from_dict(cls, kwargs) -> "JobConfig":
        """Creates a new instance of JobConfig from a dictionary.
        
        Raises
        ------
        TypeError
            If the created task configuration doesn't implement TaskConfigProtocol.
        """

        params = utils.deep_copy_dict(kwargs)

        if "job_type" not in params:
            params["job_type"] = "hadrons"
        if params["job_type"] == "hadrons":
            if "task_type" not in params:
                params["task_type"] = "lmi"

        task_type = params.get("task_type", None)

        task_params = params.get("tasks", {})
        task_builder: t.Callable = tasks.get_task_factory(params["job_type"], task_type)
        task_config = task_builder(task_params)
        
        # Validate that the task configuration implements TaskConfigProtocol
        if not isinstance(task_config, TaskConfigProtocol):
            raise TypeError(
                f"Task config {type(task_config)} from {params['job_type']}.{task_type} "
                f"does not implement TaskConfigProtocol. "
                f"Expected protocol methods: input_params, processing_params, "
                f"catalog_files, bad_files"
            )
        
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


def input_params(
    job_config: JobConfig, *args, **kwargs
) -> Union[t.Tuple[List[Dict], Optional[List[str]]], Dict[str, Dict]]:
    """Generate input parameters using the TaskConfigProtocol interface.

    .. deprecated::
        Use job_config.input_params() directly instead.

    Parameters
    ----------
    job_config : JobConfig
        Job configuration containing task configuration.
    *args, **kwargs
        Additional arguments passed to the task's input_params method.

    Returns
    -------
    Union[Tuple[List[Dict], Optional[List[str]]], Dict[str, Dict]]
        Input parameters for job execution.

    Raises
    ------
    TypeError
        If the task configuration doesn't implement TaskConfigProtocol.
    """
    warnings.warn(
        "config.input_params() is deprecated. Use job_config.input_params() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return job_config.input_params(*args, **kwargs)


def processing_params(job_config: JobConfig, *args, **kwargs) -> Dict:
    """Generate processing parameters using the TaskConfigProtocol interface.

    .. deprecated::
        Use job_config.processing_params() directly instead.

    Parameters
    ----------
    job_config : JobConfig
        Job configuration containing task configuration.
    *args, **kwargs
        Additional arguments passed to the task's processing_params method.

    Returns
    -------
    Dict
        Processing configuration parameters.

    Raises
    ------
    TypeError
        If the task configuration doesn't implement TaskConfigProtocol.
    """
    warnings.warn(
        "config.processing_params() is deprecated. Use job_config.processing_params() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return job_config.processing_params(*args, **kwargs)


def bad_files(job_config: JobConfig, *args, **kwargs) -> List[str]:
    """Identify bad files using the TaskConfigProtocol interface.

    .. deprecated::
        Use job_config.bad_files() directly instead.

    Parameters
    ----------
    job_config : JobConfig
        Job configuration containing task configuration.
    *args, **kwargs
        Additional arguments passed to the task's bad_files method.

    Returns
    -------
    List[str]
        List of problematic file paths.

    Raises
    ------
    TypeError
        If the task configuration doesn't implement TaskConfigProtocol.
    """
    warnings.warn(
        "config.bad_files() is deprecated. Use job_config.bad_files() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return job_config.bad_files(*args, **kwargs)


def catalog_files(job_config: JobConfig, *args, **kwargs) -> pd.DataFrame:
    """Generate file catalog using the TaskConfigProtocol interface.

    .. deprecated::
        Use job_config.catalog_files() directly instead.

    Parameters
    ----------
    job_config : JobConfig
        Job configuration containing task configuration.
    *args, **kwargs
        Additional arguments passed to the task's catalog_files method.

    Returns
    -------
    pd.DataFrame
        File catalog with metadata.

    Raises
    ------
    TypeError
        If the task configuration doesn't implement TaskConfigProtocol.
    """
    warnings.warn(
        "config.catalog_files() is deprecated. Use job_config.catalog_files() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return job_config.catalog_files(*args, **kwargs)
