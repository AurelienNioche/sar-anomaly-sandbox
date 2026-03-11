import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train anomaly detection model on SAR data")
    parser.add_argument("--config", type=str, default=None, help="Path to experiment config")
    parser.add_argument(
        "--output", type=str, default="checkpoints", help="Output directory for checkpoints"
    )
    parser.parse_args()
    raise NotImplementedError("Training loop not yet implemented")


if __name__ == "__main__":
    main()
