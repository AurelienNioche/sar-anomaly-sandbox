import torch

from src.data.generators.telemetry import (
    ANOMALY_TYPES,
    TelemetryGenerator,
    TelemetryGeneratorConfig,
)


def _default_gen(**kwargs) -> TelemetryGenerator:
    cfg = TelemetryGeneratorConfig(n_channels=4, n_timesteps=500, seed=0, **kwargs)
    return TelemetryGenerator(cfg)


def test_output_shapes() -> None:
    gen = _default_gen()
    telemetry, labels = gen.generate()
    assert telemetry.shape == (500, 4)
    assert labels.shape == (500,)


def test_labels_binary() -> None:
    _, labels = _default_gen().generate()
    assert set(labels.tolist()).issubset({0, 1})


def test_anomaly_ratio_approximate() -> None:
    cfg = TelemetryGeneratorConfig(
        n_channels=4, n_timesteps=2000, anomaly_ratio=0.1, seed=1
    )
    _, labels = TelemetryGenerator(cfg).generate()
    ratio = labels.float().mean().item()
    assert 0.03 < ratio < 0.25, f"Anomaly ratio {ratio:.3f} outside expected range"


def test_reproducibility() -> None:
    cfg = TelemetryGeneratorConfig(n_channels=4, n_timesteps=200, seed=42)
    t1, l1 = TelemetryGenerator(cfg).generate()
    t2, l2 = TelemetryGenerator(cfg).generate()
    assert torch.allclose(t1, t2)
    assert torch.equal(l1, l2)


def test_each_anomaly_type_injectable() -> None:
    for atype in ANOMALY_TYPES:
        cfg = TelemetryGeneratorConfig(
            n_channels=4,
            n_timesteps=500,
            anomaly_ratio=0.1,
            anomaly_types=[atype],
            seed=0,
        )
        _, labels = TelemetryGenerator(cfg).generate()
        assert labels.sum().item() > 0, f"No anomalies injected for type '{atype}'"


def test_float32_dtype() -> None:
    telemetry, _ = _default_gen().generate()
    assert telemetry.dtype == torch.float32


def test_channel_names_length() -> None:
    from src.data.generators.telemetry import CHANNEL_NAMES
    assert len(CHANNEL_NAMES) >= 7
