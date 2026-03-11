import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection model on SAR data")
    parser.add_argument("--config", type=str, default=None, help="Path to experiment config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.parse_args()
    raise NotImplementedError("Evaluation loop not yet implemented")


if __name__ == "__main__":
    main()
