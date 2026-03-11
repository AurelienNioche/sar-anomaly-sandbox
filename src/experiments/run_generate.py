import argparse
from datetime import datetime
from pathlib import Path

import torch

from src.data.generators import SpeckleSARGenerator
from src.data.generators.speckle import SpeckleSARGeneratorConfig
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic SAR-like data")
    parser.add_argument("--config", type=str, default=None, help="Path to data generation config")
    parser.add_argument("--output", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples to generate")
    args = parser.parse_args()

    default_config_path = (
        Path(__file__).parent.parent.parent / "configs" / "data" / "synthetic.yaml"
    )
    config_path = Path(args.config) if args.config else default_config_path
    cfg = load_config(config_path)

    gen_config = SpeckleSARGeneratorConfig(
        patch_size=cfg.get("patch_size", 64),
        n_looks=cfg.get("n_looks", 4),
        anomaly_ratio=cfg.get("anomaly_ratio", 0.1),
        anomaly_size=cfg.get("anomaly_size", 3),
        base_intensity=cfg.get("base_intensity", 1.0),
        anomaly_intensity=cfg.get("anomaly_intensity", 3.0),
        seed=cfg.get("seed"),
    )

    generator = SpeckleSARGenerator(gen_config)
    patches, labels = generator.generate(args.n_samples)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.output) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(patches, out_dir / "patches.pt")
    torch.save(labels, out_dir / "labels.pt")

    print(f"Saved {args.n_samples} samples to {out_dir}")


if __name__ == "__main__":
    main()
