"""End-to-end integration tests: generate telemetry -> fit detectors -> evaluate."""

from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score

from src.data.generators.telemetry import TelemetryGenerator, TelemetryGeneratorConfig
from src.models.baselines import CUSUMDetector, MahalanobisDetector, PerChannelZScore
from src.models.classical import IsolationForestDetector, OneClassSVMDetector
from src.utils.config import load_config

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "data" / "telemetry.yaml"


def _load_gen_config() -> TelemetryGeneratorConfig:
    cfg = load_config(CONFIG_PATH)
    return TelemetryGeneratorConfig(
        n_channels=cfg.get("n_channels", 7),
        n_timesteps=cfg.get("n_timesteps", 1000),
        noise_std=cfg.get("noise_std", 0.1),
        orbital_period_steps=cfg.get("orbital_period_steps", 200),
        anomaly_ratio=cfg.get("anomaly_ratio", 0.05),
        anomaly_types=cfg.get("anomaly_types", ["spike", "step", "ramp", "correlation_break"]),
        seed=cfg.get("seed"),
    )


def _generate() -> tuple[torch.Tensor, torch.Tensor]:
    gen_cfg = _load_gen_config()
    return TelemetryGenerator(gen_cfg).generate()


def _split(telemetry: torch.Tensor, labels: torch.Tensor):
    normal = telemetry[labels == 0]
    n_train = max(1, len(normal) // 2)
    return normal[:n_train], telemetry, labels


def _gen_spike_step_only() -> tuple[torch.Tensor, torch.Tensor]:
    """Generate data with only spike/step anomalies — suitable for z-score."""
    cfg = _load_gen_config()
    cfg.anomaly_types = ["spike", "step"]
    cfg.anomaly_ratio = 0.1
    return TelemetryGenerator(cfg).generate()


def test_zscore_pipeline_auc() -> None:
    """PerChannelZScore tested on spike/step only (correlation_break reduces z-scores)."""
    telemetry, labels = _gen_spike_step_only()
    assert labels.sum().item() > 0
    train, data, labels = _split(telemetry, labels)
    det = PerChannelZScore(window=20).fit(train)
    scores = det.score(data).numpy()
    auc = roc_auc_score(labels.numpy(), scores)
    assert auc > 0.6, f"PerChannelZScore AUC={auc:.3f} on spike/step data"


def _gen_with_types(anomaly_types: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = _load_gen_config()
    cfg.anomaly_types = anomaly_types
    cfg.anomaly_ratio = 0.1
    return TelemetryGenerator(cfg).generate()


def test_mahalanobis_pipeline_auc() -> None:
    """Mahalanobis tested on all anomaly types (multivariate, captures correlation)."""
    telemetry, labels = _generate()
    assert labels.sum().item() > 0
    train, data, labels = _split(telemetry, labels)
    det = MahalanobisDetector(window=20).fit(train)
    scores = det.score(data).numpy()
    auc = roc_auc_score(labels.numpy(), scores)
    assert auc > 0.6, f"MahalanobisDetector AUC={auc:.3f} too low"


def test_cusum_pipeline_auc() -> None:
    """CUSUM tested on a stationary process with a step anomaly at the end.

    CUSUM does not reset after an anomaly (accumulated sum stays high), so putting
    the anomaly at the tail ensures post-anomaly normal data does not corrupt the AUC.
    """
    n_t, n_c = 600, 3
    torch.manual_seed(7)
    normal = torch.randn(n_t, n_c) * 0.1
    data = normal.clone()
    data[400:, 0] += 2.0
    labels = torch.zeros(n_t, dtype=torch.long)
    labels[400:] = 1

    train = normal[:300]
    det = CUSUMDetector().fit(train)
    scores = det.score(data).numpy()
    auc = roc_auc_score(labels.numpy(), scores)
    assert auc > 0.8, f"CUSUMDetector AUC={auc:.3f} on stationary tail-step data"


def test_ml_pipeline_auc() -> None:
    telemetry, labels = _generate()
    train, data, labels = _split(telemetry, labels)

    for det in [IsolationForestDetector(window=20), OneClassSVMDetector(window=20)]:
        det.fit(train)
        scores = det.score(data).numpy()
        auc = roc_auc_score(labels.numpy(), scores)
        assert auc > 0.6, f"{type(det).__name__} AUC={auc:.3f} too low in integration test"


def test_data_io_roundtrip() -> None:
    import tempfile

    from src.visualization.dashboards.data_io import load_tensors_from_dir, save_run
    telemetry, labels = _generate()
    with tempfile.TemporaryDirectory() as tmp:
        save_run({"telemetry.pt": telemetry, "labels.pt": labels}, base_dir=tmp)
        result = load_tensors_from_dir(tmp, ("telemetry.pt", "labels.pt"))
    assert result is not None
    (tel_out, lab_out), _ = result
    assert tel_out.shape == telemetry.shape
    assert torch.equal(labels, lab_out)
