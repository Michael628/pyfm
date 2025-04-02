import logging
import os.path
import typing as t

import pandas as pd
from pydantic.dataclasses import dataclass

from python_scripts import config as c
from python_scripts import utils
from python_scripts.nanny import SubmitConfig, TaskBase
from python_scripts.nanny.config import OutfileList


@c.dataclass_with_getters
class SubmitSmearConfig(SubmitConfig):
    space: int
    series: t.Optional[str] = None
    cfg: t.Optional[str] = None

@dataclass
class SmearTask(TaskBase):
    unsmeared_file: str

def input_params(tasks: SmearTask, submit_config: SubmitSmearConfig,
                 outfile_config_list: OutfileList) -> t.Tuple[str,None]:

    submit_conf_dict = submit_config.string_dict()

    lat = tasks.unsmeared_file.format(**submit_conf_dict)
    lat_ildg_path = outfile_config_list.gauge_links.filename
    lat_ildg_path = lat_ildg_path.format(**submit_conf_dict)
    long_ildg_path = outfile_config_list.long_links.filename
    long_ildg_path = long_ildg_path.format(**submit_conf_dict)
    fat_ildg_path = outfile_config_list.fat_links.filename
    fat_ildg_path = fat_ildg_path.format(**submit_conf_dict)
    lat_ildg = os.path.basename(lat_ildg_path)
    long_ildg = os.path.basename(long_ildg_path)
    fat_ildg = os.path.basename(fat_ildg_path)

    space = submit_config.space
    time = submit_config.time
    input_string = "\n".join([
        "prompt 0",
        f"nx {space}",
        f"ny {space}",
        f"nz {space}",
        f"nt {time}",
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
    ])

    return input_string, None

def catalog_files(task_config: SmearTask, submit_config: SubmitSmearConfig,
                  outfile_config_list: OutfileList) -> pd.DataFrame:
    outfile_configs = [outfile_config_list.gauge_links,
                       outfile_config_list.long_links,
                       outfile_config_list.fat_links]

    def build_row(filepath: str, repls: t.Dict[str, str]) -> t.Dict[str, str]:
        repls['filepath'] = filepath
        return repls

    replacements = submit_config.string_dict()

    df = []
    for outfile_config in outfile_configs:
        outfile = outfile_config.filestem + outfile_config.ext
        filekeys = utils.formatkeys(outfile)
        files = utils.process_files(
            outfile,
            processor=build_row,
            replacements={k: v for k, v in replacements.items() if k in filekeys}
        )
        dict_of_rows = {k: [file[k] for file in files] for k in files[0] if len(files) > 0}

        new_df = pd.DataFrame(dict_of_rows)
        new_df['good_size'] = outfile_config.good_size
        new_df['exists'] = new_df['filepath'].apply(os.path.exists)
        new_df['file_size'] = None
        new_df.loc[new_df['exists'], 'file_size'] = new_df[new_df['exists']]['filepath'].apply(os.path.getsize)
        df.append(new_df)

    df = pd.concat(df, ignore_index=True)

    return df


def bad_files(task_config: SmearTask, submit_config: SubmitSmearConfig,
              outfile_config_list: OutfileList) -> t.List[str]:
    df = catalog_files(task_config, submit_config,  outfile_config_list)

    return list(df[(df['file_size'] >= df['good_size']) != True]['filepath'])


def get_task_factory():
    return SmearTask.from_dict

def get_submit_factory() -> t.Callable[..., SubmitSmearConfig]:
    return SubmitSmearConfig.create
