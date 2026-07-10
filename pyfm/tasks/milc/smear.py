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
    ildg_links: Outfile
    long_links: Outfile
    fat_links: Outfile
    v5_links: Outfile


def build_input_params(config: SmearConfig) -> str:
    """Generates input paramters for smearing HISQ lattice using milc txt parameter input"""
    lat = config.v5_links.filename
    lat_ildg_path = config.ildg_links.filename
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
    """Enumerate the expected ILDG *output* files (thin/fat/long links).

    The MILC v5 input (``v5_links``) is a consumed input, not an output, so it
    is excluded from the catalog. Delegates to :func:`catalog_files` (same
    generator interface as ``gauge.create_outfile_catalog``) so the four-column
    contract and zero-files-match ``ValueError`` are inherited for free.
    """

    def generate_outfile_formatting():
        yield {}, config.ildg_links
        yield {}, config.long_links
        yield {}, config.fat_links

    return utils.io.catalog_files(generate_outfile_formatting())


def normalize_params(params: t.Dict) -> t.Dict:
    """Normalize SmearConfig input: absorb the ``_preprocessor`` slice and migrate legacy keys.

    - Translate the legacy ``gauge_links`` key onto the canonical ``ildg_links``.
    - Warn and drop the legacy ``unsmeared_file`` string; the input configuration
      for smearing is read from ``v5_links`` instead.
    """
    combined = params | params.pop("_preprocessor", {})
    if "gauge_links" in combined:
        combined["ildg_links"] = combined.pop("gauge_links")
    if "unsmeared_file" in combined:
        utils.get_logger().warning(
            "SmearConfig: ignoring legacy 'unsmeared_file'; the input "
            "configuration for smearing is read from 'v5_links'."
        )
        del combined["unsmeared_file"]
    return combined


# Register SmearConfig as the config for 'smear' task type
register_task(
    "milc_smear",
    SmearConfig,
    create_outfile_catalog,
    build_input_params,
    normalize_params=normalize_params,
)
