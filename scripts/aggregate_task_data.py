from pyfm import utils
from pyfm.nanny import aggregator
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Provide job step (matching params.yaml steps) to process all outputs for step."
    )
    parser.add_argument("-j", "--job", type=str, help="Job name", required=True)
    parser.add_argument(
        "-p",
        "--param-file",
        type=str,
        default="params.yaml",
        help="Parameter file location",
    )
    parser.add_argument(
        "-f", "--format", type=str, default="csv", help="Output file format"
    )
    parser.add_argument(
        "--logging-level", type=str, default="INFO", help="Set logging level"
    )
    parser.add_argument(
        "--average",
        action="store_true",
        help="Average over source times (if applicable)",
        default=False,
    )
    args = parser.parse_args()

    print("Job Step:", args.job)

    params = utils.io.load_param(args.param_file)
    utils.set_logging_level(args.logging_level)

    aggregator.aggregate_task_data(
        args.job, params, format=args.format, average=args.average
    )
