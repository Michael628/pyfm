import typing as t
import pandas as pd
import pyfm.processing.processor as pc
import pyfm.processing.dataio as dio
import argparse
import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument("file_paths", nargs="*", type=str, help="files to process")
    args = parser.parse_args()

    for file_path in args.file_paths:
        assert "dataframes/" in file_path, (
            "Always expects a 'dataframes' directory explicitly in path"
        )

        path_out = file_path.replace("dataframes/", "processed_dataframes/")
        df = t.cast(pd.DataFrame, pd.read_hdf(file_path))

        actions: t.Dict[str, t.Any] = {
            "real": True,
        }

        if "gamma" in df.index.names or "gamma" in df.columns:
            actions["index"] = ["series_cfg", "gamma", "t"]
        else:
            actions["index"] = ["series_cfg", "t"]

        if "tsource" in df.index.names or "tsource" in df.columns:
            actions["average"] = ["tsource"]

        df = pc.execute(df, actions=actions)

        if "seedkey" in df.columns and df["seedkey"].nunique() == 1:
            pc.execute(df, actions={"drop": ["seedkey"]})

        logging.info(f"Writing processed file to: {path_out}")
        dio.write_frame(df, path_out)
