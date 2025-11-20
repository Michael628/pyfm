#!/usr/bin/env python3

from pyfm import utils
from pyfm.nanny import audit_outfiles, check_jobs
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
    parser.add_argument("-j", "--job", type=str, help="Job name", default=None)
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show complete files as well as missing files",
        default=False,
    )
    parser.add_argument("-s", "--series", type=str, help="Config series", default=None)
    parser.add_argument("-n", "--config", type=str, help="Config number", default=None)
    args = parser.parse_args()
    yaml_params = utils.io.load_param(args.params)

    cfgno_steps = [(f"{args.series}.{args.config}", None)]

    logger = utils.get_logger()

    if args.job is not None and args.series is not None and args.config is not None:
        logger.info(
            f"Checking job {args.job} for config series: {args.series} and config number: {args.config}"
        )

        _ = audit_outfiles(
            args.job, yaml_params, args.series, args.config, verbose=args.verbose
        )
    else:
        logger.info("Starting job checker.")
        check_jobs(yaml_params)
