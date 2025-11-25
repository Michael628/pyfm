from pyfm import utils
from pyfm.nanny import nanny_loop
import argparse
import os


if __name__ == "__main__":
    # Set permissions
    os.system("umask 022")

    parser = argparse.ArgumentParser(description="Start nanny job-monitoring process.")
    parser.add_argument(
        "-p",
        "--param-file",
        type=str,
        default="params.yaml",
        help="Parameter file location",
    )
    parser.add_argument(
        "-t", "--todo", type=str, default="todo", help="todo file location"
    )
    parser.add_argument(
        "--logging-level", type=str, default="INFO", help="Set logging level"
    )
    parser.add_argument("-j", "--job", type=str, help="Job name", default=None)
    args = parser.parse_args()

    utils.set_logging_level(args.logging_level)

    nanny_loop(args.param_file, require_step=args.job)
