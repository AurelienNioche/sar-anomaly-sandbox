import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from src.models.baselines import RXDetector
from src.visualization.dashboards.sar_dashboard import (
    LABEL_NAMES,
    outcome_label,
    patch_to_display,
)


def test_patch_to_display_shape_2d() -> None:
    patch = np.random.rand(32, 32).astype(np.float32)
    out = patch_to_display(patch)
    assert out.shape == (32, 32)
    assert out.min() >= 0


def test_patch_to_display_shape_3d() -> None:
    patch = np.random.rand(1, 32, 32).astype(np.float32)
    out = patch_to_display(patch)
    assert out.shape == (32, 32)


def test_patch_to_display_clips_negative() -> None:
    patch = np.array([[[-1.0, 2.0], [0.5, 1.0]]], dtype=np.float32)
    out = patch_to_display(patch)
    assert np.all(out >= 0)


def test_patch_to_display_log_scaling() -> None:
    patch = np.ones((8, 8), dtype=np.float32)
    out = patch_to_display(patch)
    assert np.allclose(out, np.log1p(1.0))


def test_label_names() -> None:
    assert LABEL_NAMES[0] == "Normal"
    assert LABEL_NAMES[1] == "Anomaly"


def test_outcome_label_all_cases() -> None:
    assert outcome_label(1, 1) == "TP"
    assert outcome_label(0, 0) == "TN"
    assert outcome_label(0, 1) == "FP"
    assert outcome_label(1, 0) == "FN"


def _make_patches(n: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    from src.data.generators import SpeckleSARGenerator
    from src.data.generators.speckle import SpeckleSARGeneratorConfig

    config = SpeckleSARGeneratorConfig(patch_size=16, n_looks=4, anomaly_ratio=0.2, seed=0)
    return SpeckleSARGenerator(config).generate(n)


def test_detector_tab_score_shape() -> None:
    patches, labels = _make_patches()
    normal = patches[labels == 0]
    det = RXDetector().fit(normal)
    scores = det.score(patches)
    assert scores.shape == (len(patches),)


def test_detector_tab_metrics_values() -> None:
    patches, labels = _make_patches()
    normal = patches[labels == 0]
    det = RXDetector().fit(normal)
    scores = det.score(patches)
    threshold = float(scores.numpy().mean())
    preds = det.predict(patches, threshold=threshold).numpy()
    y_true = labels.numpy()
    for metric_fn in (precision_score, recall_score, f1_score):
        val = metric_fn(y_true, preds, zero_division=0)
        assert 0.0 <= val <= 1.0
