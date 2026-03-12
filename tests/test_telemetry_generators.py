import numpy as np
import torch

from src.data.generators.telemetry import (
    ANOMALY_TYPES,
    CHANNEL_NAMES,
    TelemetryGenerator,
    TelemetryGeneratorConfig,
)


def _make_gen(n_channels: int = 7, n_timesteps: int = 1000, seed: int = 0,
              **kwargs) -> TelemetryGenerator:
    cfg = TelemetryGeneratorConfig(n_channels=n_channels, n_timesteps=n_timesteps,
                                   seed=seed, **kwargs)
    return TelemetryGenerator(cfg)


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------

def test_output_shapes() -> None:
    gen = _make_gen(n_channels=4, n_timesteps=500)
    telemetry, labels = gen.generate()
    assert telemetry.shape == (500, 4)
    assert labels.shape == (500,)


def test_labels_binary() -> None:
    _, labels = _make_gen().generate()
    assert set(labels.tolist()).issubset({0, 1})


def test_anomaly_ratio_approximate() -> None:
    gen = _make_gen(n_timesteps=2000, anomaly_ratio=0.1, seed=1)
    _, labels = gen.generate()
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
        gen = _make_gen(n_channels=7, n_timesteps=500, anomaly_ratio=0.1,
                        anomaly_types=[atype])
        _, labels = gen.generate()
        assert labels.sum().item() > 0, f"No anomalies injected for type '{atype}'"


def test_float32_dtype() -> None:
    telemetry, _ = _make_gen(n_channels=4).generate()
    assert telemetry.dtype == torch.float32


def test_channel_names_length() -> None:
    assert len(CHANNEL_NAMES) >= 7


# ---------------------------------------------------------------------------
# Channel-model properties (the new design)
# ---------------------------------------------------------------------------

def test_channel_std_populated_after_generate() -> None:
    """_channel_std must be set and contain one positive value per channel."""
    gen = _make_gen()
    assert gen._channel_std is None, "_channel_std should be None before generate()"
    gen.generate()
    assert gen._channel_std is not None
    assert gen._channel_std.shape == (7,)
    assert (gen._channel_std > 0).all(), "All channel stds must be positive"


def test_channels_0_1_are_correlated() -> None:
    """power_v and batt_soc share a noise term and should be positively correlated."""
    gen = _make_gen(anomaly_ratio=0.0, seed=1)
    telemetry, _ = gen.generate()
    r = float(np.corrcoef(telemetry[:, 0].numpy(), telemetry[:, 1].numpy())[0, 1])
    assert r > 0.5, f"Channels 0-1 correlation = {r:.3f}, expected > 0.5"


def test_ou_channels_are_stationary() -> None:
    """gyro_x / gyro_y (channels 4-5) are OU — their mean should be near 0."""
    gen = _make_gen(anomaly_ratio=0.0, n_timesteps=2000, seed=2)
    telemetry, _ = gen.generate()
    for ch in [4, 5]:
        mean = float(telemetry[:, ch].mean())
        assert abs(mean) < 0.2, f"Channel {ch} mean = {mean:.3f}, expected near 0"


def test_anomaly_spike_exceeds_5sigma() -> None:
    """Spike anomalies must be ≥5σ above the channel's own std."""
    gen = _make_gen(anomaly_ratio=0.2, anomaly_types=["spike"], seed=3)
    telemetry, labels = gen.generate()
    gen_std = gen._channel_std  # set after generate()

    normal = telemetry[labels == 0]
    channel_mean = normal.mean(dim=0).numpy()

    anom_idx = (labels == 1).nonzero(as_tuple=True)[0]
    for t in anom_idx[:10]:
        row = telemetry[t].numpy()
        max_z = float(np.max(np.abs(row - channel_mean) / gen_std))
        assert max_z >= 3.0, f"Spike at t={t} had max z={max_z:.2f}, expected ≥3σ"


def test_step_anomaly_magnitude_vs_channel_std() -> None:
    """Step anomalies must shift a channel by ≥2σ of that channel's std."""
    gen = _make_gen(n_timesteps=2000, anomaly_ratio=0.15,
                    anomaly_types=["step"], seed=4,
                    anomaly_min_duration=15, anomaly_max_duration=15)
    telemetry, labels = gen.generate()
    gen_std = gen._channel_std

    normal = telemetry[labels == 0]
    channel_mean = normal.mean(dim=0).numpy()

    anom_rows = telemetry[labels == 1].numpy()
    max_z_per_anomaly = np.max(
        np.abs(anom_rows - channel_mean[np.newaxis, :]) / gen_std[np.newaxis, :],
        axis=1,
    )
    assert max_z_per_anomaly.mean() > 2.0, (
        f"Step anomaly mean max-z = {max_z_per_anomaly.mean():.2f}, expected > 2σ"
    )


def test_correlation_break_preserves_amplitude() -> None:
    """correlation_break must keep channel 1's amplitude similar to normal,
    so a univariate z-score detector cannot trivially flag it."""
    gen = _make_gen(n_timesteps=2000, anomaly_ratio=0.15,
                    anomaly_types=["correlation_break"], seed=5)
    telemetry, labels = gen.generate()

    normal_std = float(telemetry[labels == 0, 1].std())
    anomaly_std = float(telemetry[labels == 1, 1].std())
    ratio = anomaly_std / (normal_std + 1e-9)
    assert 0.3 < ratio < 3.0, (
        f"correlation_break changed channel-1 amplitude too much "
        f"(normal_std={normal_std:.3f}, anomaly_std={anomaly_std:.3f})"
    )
