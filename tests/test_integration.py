"""End-to-end integration tests: config -> generate -> RX detect -> evaluate."""

from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.data.generators import SpeckleSARGenerator
from src.data.generators.speckle import SpeckleSARGeneratorConfig
from src.models.baselines import RXDetector
from src.utils.config import load_config

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "data" / "synthetic.yaml"


def _load_config_as_gen_config() -> SpeckleSARGeneratorConfig:
    cfg = load_config(CONFIG_PATH)
    return SpeckleSARGeneratorConfig(
        patch_size=cfg.get("patch_size", 64),
        n_looks=cfg.get("n_looks", 4),
        anomaly_ratio=cfg.get("anomaly_ratio", 0.1),
        anomaly_size=cfg.get("anomaly_size", 3),
        base_intensity=cfg.get("base_intensity", 1.0),
        anomaly_intensity=cfg.get("anomaly_intensity", 5.0),
        seed=cfg.get("seed"),
    )


def test_full_pipeline_auc() -> None:
    """Generate data from the project config, fit RX detector, assert AUC > 0.9."""
    gen_config = _load_config_as_gen_config()
    gen = SpeckleSARGenerator(gen_config)
    patches, labels = gen.generate(300)

    assert labels.sum().item() > 0, "No anomaly patches generated — check anomaly_ratio in config"

    normal = patches[labels == 0]
    det = RXDetector().fit(normal)
    scores = det.score(patches).numpy()

    auc = roc_auc_score(labels.numpy(), scores)
    assert auc > 0.9, (
        f"End-to-end AUC={auc:.3f} is too low. "
        "Check anomaly_intensity in configs/data/synthetic.yaml."
    )


def test_full_pipeline_predict_f1() -> None:
    """Fit RX detector and verify F1 > 0.7 at the optimal threshold."""
    from sklearn.metrics import f1_score

    gen_config = _load_config_as_gen_config()
    gen = SpeckleSARGenerator(gen_config)
    patches, labels = gen.generate(300)

    normal = patches[labels == 0]
    det = RXDetector().fit(normal)
    scores = det.score(patches).numpy()
    y_true = labels.numpy()

    best_f1 = 0.0
    for pct in range(50, 100):
        thresh = float(np.percentile(scores, pct))
        preds = (scores > thresh).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1

    assert best_f1 > 0.7, (
        f"Best achievable F1={best_f1:.3f} is too low across all thresholds."
    )
