import os
import typing as t

import pandas as pd

from pydantic.dataclasses import dataclass

from pyfm import utils
from pyfm.domain import SimpleConfig, Outfile
from pyfm.tasks.register import register_task

# Antiperiodic time boundary, matching the grid_milc_to_ildg default applied in
# C++ when ``boundary`` is empty (see HighlyImprovedStaggeredFermionImpl setup).
DEFAULT_BOUNDARY = "1 1 1 -1"


@dataclass(frozen=True)
class GridSmearConfig(SimpleConfig):
    """Standalone config for the ``grid_milc_to_ildg`` gauge-smear task.

    Unlike the hadrons ``GaugeConfig`` (which carries mass/action behaviour for
    full LMA-style jobs), this config only describes a file-format conversion +
    smearing run: a MILC v5 configuration goes in (``v5_links``) and the thin,
    fat, and long gauge fields come out as ILDG (``gauge_links`` /
    ``fat_links`` / ``long_links``).

    The executable always reads MILC v5 and always writes all three ILDG fields,
    so it is permanently in ``format=milcv5`` / ``save_smear=true`` mode — those
    behaviours are baked into :func:`build_input_params` rather than carried as
    configurable fields.
    """

    ildg_links: Outfile
    long_links: Outfile
    fat_links: Outfile
    v5_links: Outfile


def normalize_params(params: t.Dict) -> t.Dict:
    """Preprocessing hook: route the ``_preprocessor`` (yaml ``tasks:``) slice.

    Allows callers to override the four outfile paths via the per-task input.
    """
    return params | params.pop("_preprocessor", {})


def build_input_params(config: GridSmearConfig) -> t.Dict:
    """Build the ``MilcToIldgPar`` parameters for ``grid_milc_to_ildg``.

    Maps the MILC v5 input outfile onto ``milcFile`` and the three ILDG output
    outfiles onto ``gaugeStem`` / ``gaugeFatStem`` / ``gaugeLongStem``. The
    executable appends ``.{trajectory}`` to each stem, so the bare ``filestem``
    (without the cfg extension) is used, matching ``Outfile.filename``.

    ``trajectory`` is supplied by the grid XML wrapper (from the cfg) and is
    therefore omitted here.
    """
    return dict(
        milcFile=config.v5_links.filestem,
        gaugeStem=config.ildg_links.filestem,
        gaugeFatStem=config.fat_links.filestem,
        gaugeLongStem=config.long_links.filestem,
        boundary=DEFAULT_BOUNDARY,
        exitOnChecksumMismatch="false",
        ensembleLabel="",
    )


def create_outfile_catalog(config: GridSmearConfig) -> pd.DataFrame:
    """Enumerate the expected ILDG *output* files (thin/fat/long links).

    The MILC v5 input (``v5_links``) is a consumed input, not an output, so it
    is excluded from the catalog.
    """

    def build_row(filepath: str, repls: t.Dict[str, str]) -> t.Dict[str, str]:
        repls["filepath"] = filepath
        return repls

    outfile_configs = [config.ildg_links, config.fat_links, config.long_links]

    df = []
    for outfile_config in outfile_configs:
        files = utils.io.process_files(outfile_config.filename, processor=build_row)

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

    return pd.concat(df, ignore_index=True)


# Register GridSmearConfig as the config for the 'grid_smear' task type
register_task(
    "grid_smear",
    GridSmearConfig,
    create_outfile_catalog,
    build_input_params,
    normalize_params=normalize_params,
)
