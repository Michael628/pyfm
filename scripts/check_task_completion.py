#!/usr/bin/env python3

from pyfm import utils
from pyfm.nanny import print_file_audit
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        help="Parameter file",
        required=False,
        default="params.yaml",
    )
    parser.add_argument("-j", "--job", type=str, help="Job name", required=True)
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show complete files as well as missing files",
        default=False,
    )
    parser.add_argument("-s", "--series", type=str, help="Config series", required=True)
    parser.add_argument("-n", "--config", type=str, help="Config number", required=True)
    args = parser.parse_args()
    print("Step value:", args.job)
    print("Series value:", args.series)
    print("Config value:", args.config)

    param = utils.io.load_param(args.params)

    cfgno_steps = [(f"{args.series}.{args.config}", None)]

    df = print_file_audit(
        args.job, param, args.series, args.config, verbose=args.verbose
    )
