"""End-to-end integration tests: generate telemetry -> fit detectors -> evaluate."""

from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score

from src.data.generators.telemetry import TelemetryGenerator, TelemetryGeneratorConfig
from src.models.baselines import CUSUMDetector, MahalanobisDetector, PerChannelZScore
from src.models.classical import IsolationForestDetector, OneClassSVMDetector
from src.utils.config import load_config

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "data" / "telemetry.yaml"

N_SERIES = 10
N_TRAIN = 5


def _load_gen_config() -> TelemetryGeneratorConfig:
    cfg = load_config(CONFIG_PATH)
    return TelemetryGeneratorConfig(
        n_channels=cfg.get("n_channels", 7),
        n_timesteps=cfg.get("n_timesteps", 1000),
        noise_std=cfg.get("noise_std", 0.05),
        orbital_period_steps=cfg.get("orbital_period_steps", 200),
        anomaly_ratio=cfg.get("anomaly_ratio", 0.05),
        anomaly_types=cfg.get("anomaly_types", ["spike", "step", "ramp", "correlation_break"]),
        seed=cfg.get("seed"),
    )


def _generate() -> tuple[torch.Tensor, torch.Tensor]:
    return TelemetryGenerator(_load_gen_config()).generate(n_series=N_SERIES)


def _generate_with_types(types: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = _load_gen_config()
    cfg.anomaly_types = types
    cfg.anomaly_ratio = 0.1
    return TelemetryGenerator(cfg).generate(n_series=N_SERIES)


def _train_split(telemetry: torch.Tensor) -> torch.Tensor:
    """Return training data: first N_TRAIN series concatenated, no label filtering."""
    n, t, c = telemetry.shape
    return telemetry[:N_TRAIN].reshape(N_TRAIN * t, c)


def _test_data(telemetry: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return test series (N_TRAIN..) concatenated with binary labels."""
    n, t, c = telemetry.shape
    tel_test = telemetry[N_TRAIN:].reshape((n - N_TRAIN) * t, c)
    lab_test = (labels[N_TRAIN:].reshape(-1) > 0).long()
    return tel_test, lab_test


def test_zscore_pipeline_auc() -> None:
    """PerChannelZScore on spike/step/ramp.

    Anomalies are ≥3σ of each channel's own std, well above the sinusoidal
    fluctuations. correlation_break is excluded — it keeps the same amplitude as
    normal data and is only detectable by multivariate methods.
    """
    telemetry, labels = _generate_with_types(["spike", "step", "ramp"])
    assert (labels > 0).any()
    train = _train_split(telemetry)
    tel_test, lab_test = _test_data(telemetry, labels)
    det = PerChannelZScore(window=20).fit(train)
    scores = det.score(tel_test).numpy()
    auc = roc_auc_score(lab_test.numpy(), scores)
    assert auc > 0.75, f"PerChannelZScore AUC={auc:.3f} on spike/step/ramp data"


def test_mahalanobis_pipeline_auc() -> None:
    """MahalanobisDetector on all anomaly types including correlation_break."""
    telemetry, labels = _generate()
    assert (labels > 0).any()
    train = _train_split(telemetry)
    tel_test, lab_test = _test_data(telemetry, labels)
    det = MahalanobisDetector(window=20).fit(train)
    scores = det.score(tel_test).numpy()
    auc = roc_auc_score(lab_test.numpy(), scores)
    assert auc > 0.7, f"MahalanobisDetector AUC={auc:.3f} too low"


def test_cusum_pipeline_auc() -> None:
    """CUSUM on a stationary process with a step anomaly at the end.

    CUSUM does not reset after an anomaly (accumulated sum stays high), so putting
    the anomaly at the tail ensures post-anomaly normal data does not corrupt the AUC.
    Sinusoidal baselines confuse CUSUM (periodic accumulation/reset); this test
    uses stationary data to verify the algorithm works correctly in principle.
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
    train = _train_split(telemetry)
    tel_test, lab_test = _test_data(telemetry, labels)

    for det in [IsolationForestDetector(window=20), OneClassSVMDetector(window=20)]:
        det.fit(train)
        scores = det.score(tel_test).numpy()
        auc = roc_auc_score(lab_test.numpy(), scores)
        assert auc > 0.65, f"{type(det).__name__} AUC={auc:.3f} too low in integration test"


def test_data_io_roundtrip() -> None:
    import tempfile

    from src.visualization.data_io import load_tensors_from_dir, save_run
    telemetry, labels = _generate()
    with tempfile.TemporaryDirectory() as tmp:
        save_run({"telemetry.pt": telemetry, "labels.pt": labels}, base_dir=tmp)
        result = load_tensors_from_dir(tmp, ("telemetry.pt", "labels.pt"))
    assert result is not None
    (tel_out, lab_out), _ = result
    assert tel_out.shape == telemetry.shape
    assert torch.equal(labels, lab_out)
