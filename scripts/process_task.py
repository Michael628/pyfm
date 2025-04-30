from pyfm.processing import processor
import argparse

if __name__ == "__main__":
    setup_logging("DEBUG")
    parser = argparse.ArgumentParser(
        description="Proved job step (from params.yaml) to process outputs for."
    )
    parser.add_argument("step", type=str, help="Job Step", required=False)
    args = parser.parse_args()

    if args.step:
        print("Step value:", args.step)
        result = processor.main(step)
    else:
        result = processor.main()
