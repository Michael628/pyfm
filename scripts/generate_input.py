from pyfm import utils, setup_logging
from pyfm.nanny.spawnjob import make_inputs
import argparse

if __name__ == "__main__":
    setup_logging("DEBUG")
    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument("step", type=str, help="Job Step")
    parser.add_argument("series", type=str, help="Config series")
    parser.add_argument("cfg", type=str, help="Config number")
    args = parser.parse_args()
    print("Step value:", args.step)
    print("Series value:", args.series)
    print("Config value:", args.cfg)

    param = utils.load_param("params.yaml")

    cfgno_steps = [(f"{args.series}.{args.cfg}", None)]
    make_inputs(param, args.step, cfgno_steps)
