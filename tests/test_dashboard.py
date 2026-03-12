import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from src.models.baselines import RXDetector
from src.visualization.dashboards.sar_dashboard import (
    LABEL_NAMES,
    load_patches_labels_from_dir,
    outcome_label,
    patch_to_display,
    save_run,
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


def test_load_patches_labels_from_dir_flat_files() -> None:
    """Flat patches.pt/labels.pt are used when no timestamped sub-folder exists."""
    patches = torch.ones(10, 1, 16, 16)
    labels = torch.zeros(10, dtype=torch.long)
    with tempfile.TemporaryDirectory() as tmp:
        torch.save(patches, Path(tmp) / "patches.pt")
        torch.save(labels, Path(tmp) / "labels.pt")
        result = load_patches_labels_from_dir(tmp)
    assert result is not None
    patches_out, labels_out, resolved = result
    assert patches_out.shape == (10, 1, 16, 16)
    assert labels_out.shape == (10,)
    assert resolved == Path(tmp)


def test_load_patches_labels_from_dir_prefers_subdir_over_flat() -> None:
    """Timestamped sub-folder takes priority even if flat files exist alongside it."""
    flat_patches = torch.zeros(10, 1, 16, 16)
    run_patches = torch.ones(5, 1, 8, 8)
    labels = torch.zeros(5, dtype=torch.long)
    with tempfile.TemporaryDirectory() as tmp:
        torch.save(flat_patches, Path(tmp) / "patches.pt")
        torch.save(labels, Path(tmp) / "labels.pt")
        run_dir = Path(tmp) / "2026-03-12_10-00-00"
        run_dir.mkdir()
        torch.save(run_patches, run_dir / "patches.pt")
        torch.save(labels, run_dir / "labels.pt")
        result = load_patches_labels_from_dir(tmp)
    assert result is not None
    patches_out, _, resolved = result
    assert resolved == run_dir
    assert patches_out.shape == (5, 1, 8, 8)


def test_load_patches_labels_from_dir_auto_latest() -> None:
    """When patches.pt is only in a sub-folder, the latest run is auto-selected."""
    patches = torch.ones(5, 1, 8, 8)
    labels = torch.zeros(5, dtype=torch.long)
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "2026-03-12_10-00-00"
        run_dir.mkdir()
        torch.save(patches, run_dir / "patches.pt")
        torch.save(labels, run_dir / "labels.pt")
        result = load_patches_labels_from_dir(tmp)
    assert result is not None
    patches_out, labels_out, resolved = result
    assert patches_out.shape == (5, 1, 8, 8)
    assert resolved == run_dir


def test_load_patches_labels_from_dir_missing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        result = load_patches_labels_from_dir(tmp)
    assert result is None


def test_load_patches_labels_from_dir_nonexistent() -> None:
    result = load_patches_labels_from_dir("/nonexistent/path")
    assert result is None


def test_save_run_creates_timestamped_dir() -> None:
    patches = torch.ones(6, 1, 16, 16)
    labels = torch.zeros(6, dtype=torch.long)
    with tempfile.TemporaryDirectory() as tmp:
        saved = save_run(patches, labels, base_dir=tmp)
        assert saved.exists()
        assert (saved / "patches.pt").exists()
        assert (saved / "labels.pt").exists()
        assert saved.parent == Path(tmp)


def test_save_run_data_roundtrip() -> None:
    patches = torch.rand(8, 1, 16, 16)
    labels = torch.tensor([0, 1, 0, 1, 0, 0, 1, 0], dtype=torch.long)
    with tempfile.TemporaryDirectory() as tmp:
        saved = save_run(patches, labels, base_dir=tmp)
        loaded_patches = torch.load(saved / "patches.pt", weights_only=True)
        loaded_labels = torch.load(saved / "labels.pt", weights_only=True)
    assert torch.equal(patches, loaded_patches)
    assert torch.equal(labels, loaded_labels)


def test_save_run_multiple_runs_distinct_dirs() -> None:
    patches = torch.ones(4, 1, 8, 8)
    labels = torch.zeros(4, dtype=torch.long)
    with tempfile.TemporaryDirectory() as tmp:
        d1 = save_run(patches, labels, base_dir=tmp)
        time.sleep(1.1)
        d2 = save_run(patches, labels, base_dir=tmp)
        assert d1 != d2
        assert len(list(Path(tmp).iterdir())) == 2


def test_auto_latest_picks_newest() -> None:
    """load_patches_labels_from_dir selects the most recently modified run."""
    patches = torch.ones(3, 1, 8, 8)
    labels = torch.zeros(3, dtype=torch.long)
    with tempfile.TemporaryDirectory() as tmp:
        d1 = save_run(patches, labels, base_dir=tmp)
        time.sleep(1.1)
        d2 = save_run(patches, labels, base_dir=tmp)
        result = load_patches_labels_from_dir(tmp)
    assert result is not None
    _, _, resolved = result
    assert resolved == d2
    assert resolved != d1


def test_auto_latest_empty_dir() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        assert load_patches_labels_from_dir(tmp) is None


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
