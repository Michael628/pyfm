import logging
from pyfm import setup_logging, utils
from pyfm.processing import processor, dataio
import argparse
from pyfm.nanny import config
import typing as t
import pandas as pd


def cfg_sort(cfg_list: t.List[str]) -> t.List[str]:
    """
    Sort the configuration list based on the configuration name.
    """
    return sorted(cfg_list, key=lambda x: (x.split(".")[0], int(x.split(".")[1])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine processed DataFrame h5 files into a single DataFrame file with only intersecting configurations."
    )
    parser.add_argument("outfile", type=str, help="Name for output file.")
    parser.add_argument("filestem", type=str, help="File pattern for input files.")
    args = parser.parse_args()

    logging_level = input("Enter logging level (INFO): ").strip().upper() or "INFO"
    setup_logging(logging_level)

    shared_cfgs: t.Optional[t.Set[str]] = None
    df_out: t.List[pd.DataFrame] = []

    df_out = dataio.load(args.filestem)

    for file_path, df in df_out.items():
        assert "series_cfg" in df.index.names, (
            f"`series_cfg` must be in index of {file_path}"
        )
        cfg_list = t.cast(
            t.List[str],
            df.index.get_level_values("series_cfg").drop_duplicates().to_list(),
        )
        cfg_list = cfg_sort(cfg_list)

        logging.info(f"Processing file {file_path}")
        logging.debug(f"Configs in current file: {', '.join(cfg_list)}")

        if shared_cfgs is None:
            shared_cfgs = set(cfg_list)
        else:
            new_cfgs = set(cfg_list)
            shared_cfgs = shared_cfgs & new_cfgs
            missing_cfgs = shared_cfgs - new_cfgs
            excess_cfgs = new_cfgs - shared_cfgs
            if len(excess_cfgs) > 0:
                logging.warning(
                    f"Extra configs in current file (will be excluded): {', '.join(excess_cfgs)}"
                )
            if len(missing_cfgs) > 0:
                logging.warning(
                    f"Missing configs in current file: {', '.join(missing_cfgs)}"
                )

        assert len(shared_cfgs) > 0, "No configs to return!"

    assert shared_cfgs is not None, "No configs found in file arg(s)!"
    logging.info(
        f"{len(shared_cfgs)} configs found in all files: {', '.join(shared_cfgs)}"
    )

    df_out = pd.concat(df_out.values())
    df_out = df_out[df_out.index.get_level_values("series_cfg").isin(shared_cfgs)]
    dataio.write_frame(df_out, args.outfile)
