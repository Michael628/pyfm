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
    parser.add_argument("step", type=str, help="Job Step")
    parser.add_argument(
        "--param-file", type=str, default="params.yaml", help="Parameter file location"
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

        # TODO: move "type" param to flag --format that...gets passed to processing_params?
        if out_files:
            out_type = out_files["type"]
            if out_type == "dictionary":
                filestem = out_files["filestem"]
                depth = int(out_files["depth"])
                dio.write_dict(result[key], filestem, depth)
            elif out_type == "dataframe":
                filestem = out_files["filestem"]
                dio.write_frame(result[key], filestem)
            else:
                raise NotImplementedError(f"No support for out file type {out_type}.")
