import os
import typing as t

import pandas as pd

from dataclasses import dataclass
from pyfm import utils
from pyfm.domain import SimpleConfig, Outfile
from pyfm.tasks.register import register_task


@dataclass(frozen=True)
class SmearConfig(SimpleConfig):
    time: int
    space: int
    node_geometry: str
    gauge_links: Outfile
    long_links: Outfile
    fat_links: Outfile
    unsmeared_file: str

    key: t.ClassVar[str] = "smear"


def build_input_params(config: SmearConfig) -> str:
    """Generates input paramters for smearing HISQ lattice using milc txt parameter input"""
    lat = config.unsmeared_file
    lat_ildg_path = config.gauge_links.filename
    long_ildg_path = config.long_links.filename
    fat_ildg_path = config.fat_links.filename

    lat_ildg = os.path.basename(lat_ildg_path)
    long_ildg = os.path.basename(long_ildg_path)
    fat_ildg = os.path.basename(fat_ildg_path)

    space = config.space
    time = config.time
    node_geometry = config.node_geometry
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

    return input_string


def create_outfile_catalog(config: SmearConfig) -> pd.DataFrame:
    outfile_configs = [
        config.gauge_links,
        config.long_links,
        config.fat_links,
    ]

    def build_row(filepath: str, repls: t.Dict[str, str]) -> t.Dict[str, str]:
        repls["filepath"] = filepath
        return repls

    df = []
    for outfile_config in outfile_configs:
        outfile = outfile_config.filename

        files = utils.io.process_files(outfile, processor=build_row)

        dict_of_rows = {
            k: [file[k] for file in files] for k in files[0] if len(files) > 0
        }

        new_df = (
            pd.DataFrame(dict_of_rows)
            .assign(good_size=outfile_config.good_size)
            .assign(exists=lambda df: df["filepath"].apply(os.path.exists))
            .assign(
                file_size=lambda df: df[df["exists"]]["filepath"].transform(
                    os.path.getsize
                )
            )
        )
        df.append(new_df)

    df = pd.concat(df, ignore_index=True)

    return df


# Register SmearConfig as the config for 'smear' task type
register_task(SmearConfig, create_outfile_catalog, build_input_params)
