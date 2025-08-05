from pyfm import utils, setup_logging
from pyfm.nanny.spawnjob import make_inputs
import argparse

if __name__ == "__main__":
    setup_logging("DEBUG")
    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument("-j", "--job", type=str, help="Job name")
    parser.add_argument("-s", "--series", type=str, help="Config series")
    parser.add_argument("-c", "--config", type=str, help="Config number")
    args = parser.parse_args()
    print("Step value:", args.job)
    print("Series value:", args.series)
    print("Config value:", args.config)

    param = utils.load_param("params.yaml")

    cfgno_steps = [(f"{args.series}.{args.config}", None)]
    make_inputs(param, args.job, cfgno_steps)
