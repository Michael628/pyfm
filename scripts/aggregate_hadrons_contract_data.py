import pandas as pd

from pyfm import utils
from pyfm.nanny import aggregator
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Provide file path pattern to aggregate data."
    )

    parser.add_argument(
        "-o", "--output-path", type=str, help="Output file path", required=True
    )

    parser.add_argument(
        "-t",
        "--time",
        type=int,
        help="Nt, lattice units in time direction",
        required=True,
    )

    parser.add_argument(
        "-i", "--input-path", type=str, help="Input file format", required=True
    )

    parser.add_argument(
        "-f", "--format", type=str, default="csv", help="Output file format"
    )

    parser.add_argument(
        "--logging-level", type=str, default="INFO", help="Set logging level"
    )

    args = parser.parse_args()

    utils.set_logging_level(args.logging_level)

    # h5_datasets = {"{gamma}": "/{gamma}_0_{gamma}/correlator"}
    h5_datasets = {"{gamma}": "correlator"}

    array_params = {
        "order": ["t"],
        "labels": {"t": f"0..{args.time - 1}"},
    }

    def drop_extra_gammas(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        assert (
            "gamma" in df.index.names
        ), "Expecting 'gamma' to be a valid index at this stage."

        gamma_cols = [k for k in df.columns if "gamma" in k]
        return df.drop(columns=gamma_cols)

    agg_params = {
        "run": ["hadrons_contract"],
        "hadrons_contract": {
            "logging_level": args.logging_level,
            "actions": {
                "preprocess_custom": drop_extra_gammas,
                "index": ["series_cfg", "gamma", "t"],
                "real": True,
            },
            "load_files": {
                "filestem": args.input_path,
                "regex": {"series": "[a-z]", "cfg": "[0-9]+"},
                "wildcard_fill": True,
                "name": "gamma",
                "datasets": h5_datasets,
                **array_params,
            },
            "out_files": {"filestem": args.output_path},
        },
    }

    aggregator.aggregate_data(agg_params, format=args.format)
