import typing as t
from typing import Union, List, Optional, Dict

from pydantic.dataclasses import dataclass
import pandas as pd
import pyfm
from pyfm import utils
import os


# ============Outfile Parameters===========
@dataclass
class Outfile:
    filestem: str
    ext: str
    good_size: int

    @property
    def filename(self) -> str:
        return self.filestem + self.ext

    @classmethod
    def create(
        cls,
        file_path: str,
        file_label: str,
        filestem: str,
        good_size: t.Union[str, int],
    ) -> "Outfile":
        def get_extension(fname: str) -> str:
            extensions = {
                "cfg": ".{cfg}",
                "cfg_bin": ".{cfg}.bin",
                "cfg_bin_multi": ".{cfg}/v{eig_index}.bin",
                "cfg_h5": ".{cfg}.h5",
                "cfg_gamma_h5": ".{cfg}/{gamma}_0_0_0.h5",
                "cfg_pickle": ".{cfg}.p",
            }

            if fname.endswith("links"):
                return extensions["cfg"]
            if fname.startswith("meson"):
                return extensions["cfg_gamma_h5"]
            if fname == "eig" or fname.startswith("a2a_vec"):
                return extensions["cfg_bin"]
            if fname == "contract":
                return extensions["cfg_pickle"]
            if fname.endswith("modes") or fname == "eval":
                return extensions["cfg_h5"]
            if fname == "eigdir":
                return extensions["cfg_bin_multi"]
            raise ValueError(f"No outfile definition for {fname}.")

        params = {
            "filestem": str(os.path.join(file_path, filestem)),
            "ext": get_extension(file_label),
            "good_size": good_size,
        }

        return cls(**params)


@dataclass
class TaskBase:
    """Base class for task configurations implementing TaskConfigProtocol.
    
    This class provides default implementations that delegate to the legacy
    module-based dispatch system for backward compatibility. Subclasses should
    override these methods to provide direct implementations.
    """
    
    @classmethod
    def from_dict(cls, kwargs: t.Dict[str, t.Any]) -> "TaskBase":
        return cls(**kwargs)
    
    def input_params(self, submit_config: "SubmitConfig") -> Union[
        t.Tuple[List[Dict], Optional[List[str]]], 
        Dict[str, Dict]
    ]:
        """Default implementation delegating to module-based dispatch.
        
        Subclasses should override this method to provide direct implementations.
        """
        from pyfm.nanny import tasks
        # Try to get job_type and task_type from the instance
        job_type = getattr(self, 'job_type', 'hadrons')
        task_type = getattr(self, 'task_type', 'lmi')
        return tasks.input_params(job_type, task_type, self, submit_config)
    
    def processing_params(self, submit_config: "SubmitConfig") -> Dict:
        """Default implementation delegating to module-based dispatch.
        
        Subclasses should override this method to provide direct implementations.
        """
        from pyfm.nanny import tasks
        job_type = getattr(self, 'job_type', 'hadrons')
        task_type = getattr(self, 'task_type', 'lmi')
        return tasks.processing_params(job_type, task_type, self, submit_config)
    
    def catalog_files(self, submit_config: "SubmitConfig") -> pd.DataFrame:
        """Default implementation delegating to module-based dispatch.
        
        Subclasses should override this method to provide direct implementations.
        """
        from pyfm.nanny import tasks
        job_type = getattr(self, 'job_type', 'hadrons')
        task_type = getattr(self, 'task_type', 'lmi')
        return tasks.catalog_files(job_type, task_type, self, submit_config)
    
    def bad_files(self, submit_config: "SubmitConfig") -> List[str]:
        """Default implementation delegating to module-based dispatch.
        
        Subclasses should override this method to provide direct implementations.
        """
        from pyfm.nanny import tasks
        job_type = getattr(self, 'job_type', 'hadrons')
        task_type = getattr(self, 'task_type', 'lmi')
        return tasks.bad_files(job_type, task_type, self, submit_config)


@dataclass
class SubmitConfig(pyfm.ConfigBase):
    ens: str
    time: int
    _files: t.Dict[str, Outfile]

    @property
    def files(self) -> t.Dict[str, Outfile]:
        return self._files

    @classmethod
    def create(cls, **kwargs) -> "SubmitConfig":
        params = utils.deep_copy_dict(kwargs)
        if files := params.pop("files", {}):
            home = files.pop("home")
            for k, v in files.items():
                files[k] = Outfile.create(
                    file_path=home,
                    file_label=k,
                    filestem=v["filestem"],
                    good_size=v["good_size"],
                )
        params["_files"] = files
        return super().create(**params)
