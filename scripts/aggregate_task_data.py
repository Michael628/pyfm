from pyfm import setup_logging, utils
from pyfm.nanny import config
from pyfm.processing import processor as pc
from pyfm.processing import dataio as dio
import argparse
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Provide job step (matching params.yaml steps) to process all outputs for step."
    )
    parser.add_argument("-s", "--step", type=str, help="Job Step")
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
    args = parser.parse_args()

    print("Step value:", args.step)

    params = utils.load_param(args.param_file)
    job_config = config.get_job_config(params, args.step)
    submit_config = config.get_submit_config(params, job_config)
    proc_params = config.processing_params(job_config, submit_config)

    setup_logging(args.logging_level)

    result = {}
    for key in proc_params["run"]:
        run_params = proc_params[key]

        result[key] = pd.concat(dio.load(**run_params["load_files"]).values())
        actions = run_params.get("actions", {})
        out_files = run_params.get("out_files", {})
        index = out_files.pop("index", None)

        if index:
            actions.update({"index": index})

        if "actions" in run_params:
            result[key] = pc.execute(result[key], run_params["actions"])

        keys = utils.format_keys(out_files["filestem"])

        if "format" in keys:
            result[key]["format"] = args.format
        if out_files:
            dio.write(result[key], format=args.format, **out_files)
