import numpy as np

from src.visualization.dashboards.sar_dashboard import LABEL_NAMES, patch_to_display


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
