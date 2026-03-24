from pyfm import utils
from pyfm.nanny import nanny_loop, get_nanny_config, get_job_config, submit_job
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
        "-i", "--input", type=str, help="todo file location", required=True
    )
    parser.add_argument(
        "--logging-level", type=str, default="INFO", help="Set logging level"
    )
    parser.add_argument("-j", "--job", type=str, help="Job name", required=True)
    args = parser.parse_args()

    utils.set_logging_level(args.logging_level)

    yaml_params = utils.io.load_param(args.param_file)

    nanny_config = get_nanny_config(yaml_params)

    job_config = get_job_config(args.job, yaml_params)

    os.environ["INPUTLIST"] = args.input
    submit_job(nanny_config, job_config, 1)
