import typing as t

from pydantic.dataclasses import dataclass
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
    @classmethod
    def from_dict(cls, kwargs: t.Dict[str, t.Any]) -> "TaskBase":
        return cls(**kwargs)


@dataclass
class SubmitConfig(pyfm.ConfigBase):
    ens: str
    time: int
    files: t.Dict[str, Outfile]

    @classmethod
    def create(cls, **kwargs) -> "SubmitConfig":
        params = utils.deep_copy_dict(kwargs)
        if files := params.get("files", {}):
            home = files.pop("home")
            for k, v in files.items():
                files[k] = Outfile.create(
                    file_path=home,
                    file_label=k,
                    filestem=v["filestem"],
                    good_size=v["good_size"],
                )
        params["files"] = files
        return super().create(**params)
