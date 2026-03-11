import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic SAR-like data")
    parser.add_argument("--config", type=str, default=None, help="Path to data generation config")
    parser.add_argument("--output", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples to generate")
    parser.parse_args()
    raise NotImplementedError("Synthetic data generation not yet implemented")


if __name__ == "__main__":
    main()
