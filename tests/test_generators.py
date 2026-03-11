import torch

from src.data.generators import SpeckleSARGenerator
from src.data.generators.speckle import SpeckleSARGeneratorConfig


def test_generator_shapes() -> None:
    config = SpeckleSARGeneratorConfig(
        patch_size=32, n_looks=4, anomaly_ratio=0.0, seed=42
    )
    gen = SpeckleSARGenerator(config)
    patches, labels = gen.generate(10)
    assert patches.shape == (10, 1, 32, 32)
    assert labels.shape == (10,)


def test_labels_binary() -> None:
    config = SpeckleSARGeneratorConfig(
        patch_size=32, n_looks=4, anomaly_ratio=0.5, seed=42
    )
    gen = SpeckleSARGenerator(config)
    _, labels = gen.generate(100)
    assert set(labels.tolist()).issubset({0, 1})


def test_anomaly_ratio_approximate() -> None:
    config = SpeckleSARGeneratorConfig(
        patch_size=32, n_looks=4, anomaly_ratio=0.2, seed=42
    )
    gen = SpeckleSARGenerator(config)
    _, labels = gen.generate(1000)
    ratio = labels.float().mean().item()
    assert 0.1 <= ratio <= 0.35


def test_reproducibility() -> None:
    config = SpeckleSARGeneratorConfig(
        patch_size=16, n_looks=4, anomaly_ratio=0.0, seed=123
    )
    gen1 = SpeckleSARGenerator(config)
    p1, _ = gen1.generate(5)

    gen2 = SpeckleSARGenerator(config)
    p2, _ = gen2.generate(5)

    assert torch.allclose(p1, p2)
