import logging
import os.path
import typing as t
from typing import Union, List, Optional, Dict, Tuple

import pandas as pd
from pydantic.dataclasses import dataclass

from pyfm import config as c
from pyfm import utils
from pyfm.nanny import SubmitConfig, TaskBase
from pyfm.nanny.registry import register_task, register_submit_config


@register_submit_config("smear")
@c.dataclass_with_getters
class SubmitSmearConfig(SubmitConfig):
    space: int
    node_geometry: str
    series: t.Optional[str] = None
    cfg: t.Optional[str] = None


@register_task("smear")
@dataclass
class SmearTask(TaskBase):
    unsmeared_file: str
    
    def input_params(self, submit_config: SubmitSmearConfig) -> Tuple[str, None]:
        """Generate input parameters for smear job execution.
        
        Parameters
        ----------
        submit_config : SubmitSmearConfig
            Configuration parameters for submitted job.
            
        Returns
        -------
        Tuple[str, None]
            Input string for smear execution and None for schedule.
        """
        return input_params(self, submit_config)
    
    def processing_params(self, submit_config: SubmitSmearConfig) -> Dict:
        """Generate processing parameters for smear data analysis.
        
        Note: Smear tasks typically don't have processing parameters.
        
        Parameters
        ----------
        submit_config : SubmitSmearConfig
            Configuration parameters for submitted job.
            
        Returns
        -------
        Dict
            Empty processing configuration.
        """
        return {}
    
    def catalog_files(self, submit_config: SubmitSmearConfig) -> pd.DataFrame:
        """Generate file catalog for smear outputs.
        
        Parameters
        ----------
        submit_config : SubmitSmearConfig
            Configuration parameters for submitted job.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: filepath, exists, file_size, good_size
        """
        return catalog_files(self, submit_config)
    
    def bad_files(self, submit_config: SubmitSmearConfig) -> List[str]:
        """Identify incomplete or corrupted smear files.
        
        Parameters
        ----------
        submit_config : SubmitSmearConfig
            Configuration parameters for submitted job.
            
        Returns
        -------
        List[str]
            List of problematic file paths.
        """
        return bad_files(self, submit_config)


def input_params(
    tasks: SmearTask, submit_config: SubmitSmearConfig
) -> t.Tuple[str, None]:
    submit_conf_dict = submit_config.string_dict()

    outfile_dict = submit_config.files
    lat = tasks.unsmeared_file.format(**submit_conf_dict)
    lat_ildg_path = outfile_dict["gauge_links"].filename
    lat_ildg_path = lat_ildg_path.format(**submit_conf_dict)
    long_ildg_path = outfile_dict["long_links"].filename
    long_ildg_path = long_ildg_path.format(**submit_conf_dict)
    fat_ildg_path = outfile_dict["fat_links"].filename
    fat_ildg_path = fat_ildg_path.format(**submit_conf_dict)
    lat_ildg = os.path.basename(lat_ildg_path)
    long_ildg = os.path.basename(long_ildg_path)
    fat_ildg = os.path.basename(fat_ildg_path)

    space = submit_config.space
    time = submit_config.time
    node_geometry = submit_config.node_geometry
    input_string = "\n".join(
        [
            "prompt 0",
            f"nx {space}",
            f"ny {space}",
            f"nz {space}",
            f"nt {time}",
            f"node_geometry {node_geometry}",
            f"ionode_geometry {node_geometry}",
            "iseed 1234",
            f"reload_parallel {lat}",
            "u0   1",
            f"save_serial_ildg {lat_ildg_path}",
            f"ILDG_LFN {lat_ildg}",
            "coordinate_origin 0 0 0 0",
            "time_bc antiperiodic",
            f"save_serial_ildg {long_ildg_path}",
            f"ILDG_LFN {long_ildg}",
            f"save_serial_ildg {fat_ildg_path}",
            f"ILDG_LFN {fat_ildg}",
            "withKSphases 1",
        ]
    )

    return input_string, None


def catalog_files(
    task_config: SmearTask, submit_config: SubmitSmearConfig
) -> pd.DataFrame:
    outfile_dict = submit_config.files
    outfile_configs = [
        outfile_dict["gauge_links"],
        outfile_dict["long_links"],
        outfile_dict["fat_links"],
    ]

    def build_row(filepath: str, repls: t.Dict[str, str]) -> t.Dict[str, str]:
        repls["filepath"] = filepath
        return repls

    replacements = submit_config.string_dict()

    df = []
    for outfile_config in outfile_configs:
        outfile = outfile_config.filestem + outfile_config.ext
        filekeys = utils.format_keys(outfile)
        files = utils.process_files(
            outfile,
            processor=build_row,
            replacements={k: v for k, v in replacements.items() if k in filekeys},
        )
        dict_of_rows = {
            k: [file[k] for file in files] for k in files[0] if len(files) > 0
        }

        new_df = pd.DataFrame(dict_of_rows)
        new_df["good_size"] = outfile_config.good_size
        new_df["exists"] = new_df["filepath"].apply(os.path.exists)
        new_df["file_size"] = None
        new_df.loc[new_df["exists"], "file_size"] = new_df[new_df["exists"]][
            "filepath"
        ].apply(os.path.getsize)
        df.append(new_df)

    df = pd.concat(df, ignore_index=True)

    return df


def bad_files(task_config: SmearTask, submit_config: SubmitSmearConfig) -> t.List[str]:
    df = catalog_files(task_config, submit_config)

    return list(df[(df["file_size"] >= df["good_size"]) != True]["filepath"])


def processing_params(task_config: SmearTask, submit_config: SubmitSmearConfig) -> Dict:
    """Generate processing parameters for smear tasks.
    
    Note: Smear tasks typically don't require post-processing.
    
    Returns
    -------
    Dict
        Empty processing configuration.
    """
    return {}


# Factory functions removed - now handled by plugin registry in tasks/__init__.py
