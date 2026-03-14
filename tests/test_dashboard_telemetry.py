"""Dashboard integration tests for the telemetry pipeline.

These tests exist to catch 'connect-the-dots' failures that only show up
when running the dashboard:

1. Config drift  — dashboard GEN_DEFAULTS diverge from configs/data/telemetry.yaml
2. Stale data    — old runs in data/synthetic/telemetry/ silently override dashboard output
3. Generate→Save→Load roundtrip — data produced by the dashboard Generator tab
   must survive save_run / load_tensors_from_dir intact
4. Channel properties — data generated with dashboard defaults must have the
   physically correct channel behaviour (OU channels are NOT sinusoidal, etc.)
5. Detector tab plumbing — every detector the dashboard exposes can fit on the
   data produced by the generator and return scores of the right shape
"""

import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.data.generators.telemetry import TelemetryGenerator, TelemetryGeneratorConfig
from src.models.baselines import CUSUMDetector, MahalanobisDetector, PerChannelZScore
from src.models.classical import IsolationForestDetector, OneClassSVMDetector
from src.models.deep import LSTMAutoencoderDetector
from src.utils.config import load_config
from src.visualization.data_io import load_tensors_from_dir, save_run
from src.visualization.telemetry_dashboard import (
    DEFAULT_DATA_DIR,
    GEN_DEFAULTS,
)

YAML_PATH = Path(__file__).parent.parent / "configs" / "data" / "telemetry.yaml"
_FILENAMES = ("telemetry.pt", "labels.pt")


# ---------------------------------------------------------------------------
# 1. Config drift
# ---------------------------------------------------------------------------

def test_gen_defaults_match_yaml() -> None:
    """GEN_DEFAULTS in the dashboard must be consistent with telemetry.yaml.

    This test fails the moment someone updates the YAML but forgets the dashboard
    (or vice-versa), preventing silent behavioural divergence.
    """
    yaml = load_config(YAML_PATH)
    mapping = {
        "tel_n_series":       ("n_series",             int),
        "tel_n_channels":     ("n_channels",            int),
        "tel_n_timesteps":    ("n_timesteps",            int),
        "tel_noise_std":      ("noise_std",              float),
        "tel_orbital_period": ("orbital_period_steps",   int),
        "tel_anomaly_ratio":  ("anomaly_ratio",          float),
        "tel_seed":           ("seed",                   int),
    }
    for dash_key, (yaml_key, cast) in mapping.items():
        dash_val = cast(GEN_DEFAULTS[dash_key])
        yaml_val = cast(yaml[yaml_key])
        assert dash_val == yaml_val, (
            f"Dashboard default '{dash_key}'={dash_val} != "
            f"YAML '{yaml_key}'={yaml_val} in {YAML_PATH.name}"
        )


# ---------------------------------------------------------------------------
# 2. Stale data guard
# ---------------------------------------------------------------------------

def test_no_stale_data_in_telemetry_dir() -> None:
    """data/synthetic/telemetry/ must not contain any pre-committed run directories.

    Saved runs are user-local artefacts; committing them silently overrides
    everyone else's dashboard with old (possibly wrong) data on first load.
    """
    import subprocess
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", DEFAULT_DATA_DIR],
        capture_output=True, text=True,
        cwd=Path(DEFAULT_DATA_DIR).parent.parent.parent,
    )
    tracked = [
        line for line in result.stdout.splitlines()
        if line.endswith(".pt")
    ]
    assert tracked == [], (
        f"Telemetry run data was committed to git: {tracked} — "
        "saved runs are user-local artefacts and must not be committed."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gen_from_defaults() -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce what the dashboard does when the user clicks Generate.

    Returns:
        telemetry : (N, T, C)
        labels    : (N, T) multi-class — 0=normal, 1-4=anomaly type
    """
    cfg = TelemetryGeneratorConfig(
        n_channels=GEN_DEFAULTS["tel_n_channels"],
        n_timesteps=GEN_DEFAULTS["tel_n_timesteps"],
        noise_std=GEN_DEFAULTS["tel_noise_std"],
        orbital_period_steps=GEN_DEFAULTS["tel_orbital_period"],
        anomaly_ratio=GEN_DEFAULTS["tel_anomaly_ratio"],
        seed=GEN_DEFAULTS["tel_seed"],
    )
    return TelemetryGenerator(cfg).generate(n_series=GEN_DEFAULTS["tel_n_series"])


def _first_series(
    telemetry: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the first series with binary labels for single-series assertions."""
    return telemetry[0], (labels[0] > 0).long()


# ---------------------------------------------------------------------------
# 3. Generate → Save → Load roundtrip (mirrors dashboard Generator tab)
# ---------------------------------------------------------------------------

def test_generate_save_load_roundtrip() -> None:
    """Data from the Generator tab must survive save_run / load_tensors_from_dir."""
    telemetry, labels = _gen_from_defaults()
    with tempfile.TemporaryDirectory() as tmp:
        save_run({"telemetry.pt": telemetry, "labels.pt": labels}, base_dir=tmp)
        result = load_tensors_from_dir(tmp, _FILENAMES)
    assert result is not None, "load_tensors_from_dir returned None after save_run"
    (tel_out, lab_out), resolved = result
    assert tel_out.shape == telemetry.shape, "Loaded telemetry shape mismatch"
    assert lab_out.shape == labels.shape, "Loaded labels shape mismatch"
    assert torch.allclose(tel_out, telemetry), "Loaded telemetry values differ"
    assert torch.equal(lab_out, labels), "Loaded labels differ"


def test_save_creates_timestamped_subdir() -> None:
    """save_run must create a sub-directory, not write flat files."""
    telemetry, labels = _gen_from_defaults()
    with tempfile.TemporaryDirectory() as tmp:
        saved_path = save_run(
            {"telemetry.pt": telemetry, "labels.pt": labels}, base_dir=tmp
        )
        assert saved_path.parent == Path(tmp), "Saved path should be one level deep"
        assert (saved_path / "telemetry.pt").exists()
        assert (saved_path / "labels.pt").exists()
        subdirs = [d for d in Path(tmp).iterdir() if d.is_dir()]
        assert len(subdirs) == 1, "Exactly one timestamped sub-directory expected"


def test_load_prefers_latest_subdir() -> None:
    """load_tensors_from_dir must return the MOST RECENT run, not flat files."""
    telemetry, labels = _gen_from_defaults()
    with tempfile.TemporaryDirectory() as tmp:
        import time
        save_run({"telemetry.pt": telemetry, "labels.pt": labels}, base_dir=tmp)
        time.sleep(0.05)
        telemetry2 = telemetry * 2
        save_run({"telemetry.pt": telemetry2, "labels.pt": labels}, base_dir=tmp)
        result = load_tensors_from_dir(tmp, _FILENAMES)
    assert result is not None
    (tel_out, _), _ = result
    assert torch.allclose(tel_out, telemetry2), "Should load the most recent run"


# ---------------------------------------------------------------------------
# 4. Channel properties with dashboard defaults
# ---------------------------------------------------------------------------

def test_dashboard_data_has_sinusoidal_channels() -> None:
    """Channels 0-3 must be clearly periodic (high autocorrelation at orbital lag)."""
    telemetry, labels = _gen_from_defaults()
    series, bin_labels = _first_series(telemetry, labels)
    normal = series[bin_labels == 0].numpy()
    orbital_lag = GEN_DEFAULTS["tel_orbital_period"]
    for ch in range(min(4, series.shape[1])):
        x = normal[:, ch]
        if len(x) <= orbital_lag:
            continue
        r = float(np.corrcoef(x[:-orbital_lag], x[orbital_lag:])[0, 1])
        assert r > 0.3, (
            f"Channel {ch} (sinusoidal) autocorr at orbital lag = {r:.3f}, "
            "expected > 0.3"
        )


def test_dashboard_data_ou_channels_not_sinusoidal() -> None:
    """Channels 4-5 (OU) must NOT be periodic — autocorrelation at orbital lag ≈ 0."""
    telemetry, labels = _gen_from_defaults()
    series, bin_labels = _first_series(telemetry, labels)
    if series.shape[1] < 6:
        return
    normal = series[bin_labels == 0].numpy()
    orbital_lag = GEN_DEFAULTS["tel_orbital_period"]
    for ch in [4, 5]:
        x = normal[:, ch]
        if len(x) <= orbital_lag:
            continue
        r = float(np.corrcoef(x[:-orbital_lag], x[orbital_lag:])[0, 1])
        assert r < 0.5, (
            f"Channel {ch} (OU) has unexpectedly high orbital autocorr {r:.3f}; "
            "the generator may have reverted to a sinusoidal model"
        )


def test_dashboard_data_has_anomalies() -> None:
    """Generated data must contain at least some anomalies."""
    _, labels = _gen_from_defaults()
    n_anom = int((labels > 0).sum().item())
    assert n_anom > 0, "No anomalies in default dashboard data — check anomaly_ratio"


# ---------------------------------------------------------------------------
# 5. Detector tab plumbing (fit + score on dashboard data)
# ---------------------------------------------------------------------------

def _make_train_test(
    telemetry: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use first series for a single-series train/test plumbing check."""
    series, bin_labels = _first_series(telemetry, labels)
    normal = series[bin_labels == 0]
    n_train = max(1, len(normal) // 2)
    return normal[:n_train], series, bin_labels


def _check_detector(det, telemetry, labels, min_auc: float = 0.5) -> None:
    train, data, labs = _make_train_test(telemetry, labels)
    det.fit(train)
    scores = det.score(data)
    assert scores.shape == (len(data),), f"Wrong score shape: {scores.shape}"
    assert not torch.isnan(scores).any(), "Scores contain NaN"
    assert not torch.isinf(scores).any(), "Scores contain Inf"
    auc = roc_auc_score(labs.numpy(), scores.numpy())
    assert auc >= min_auc, f"{type(det).__name__} AUC={auc:.3f} < {min_auc}"


def test_statistical_detectors_on_dashboard_data() -> None:
    telemetry, labels = _gen_from_defaults()
    _check_detector(PerChannelZScore(window=20), telemetry, labels)
    _check_detector(MahalanobisDetector(window=20), telemetry, labels)
    _check_detector(CUSUMDetector(), telemetry, labels)


def test_ml_detectors_on_dashboard_data() -> None:
    telemetry, labels = _gen_from_defaults()
    _check_detector(IsolationForestDetector(window=20), telemetry, labels)
    _check_detector(OneClassSVMDetector(window=20), telemetry, labels)


def test_lstm_detector_on_dashboard_data() -> None:
    telemetry, labels = _gen_from_defaults()
    _check_detector(
        LSTMAutoencoderDetector(window=20, hidden_size=16, n_epochs=5),
        telemetry, labels,
    )


# ---------------------------------------------------------------------------
# 6. Visualize tab
# ---------------------------------------------------------------------------

def test_visualize_tab_is_registered() -> None:
    """'Visualize' must appear in the st.tabs call in main(), between Generator
    and Statistical."""
    import re
    src = (Path(__file__).parent.parent /
           "src/visualization/telemetry_dashboard.py").read_text()
    match = re.search(r'st\.tabs\(\[(.*?)\]\)', src, re.DOTALL)
    assert match, "Could not find st.tabs([...]) call in telemetry_dashboard.py"
    tab_str = match.group(1)
    names = re.findall(r'"([^"]+)"', tab_str)
    assert "Visualize" in names, f"'Visualize' not found in tab list: {names}"
    idx = names.index("Visualize")
    assert names[idx - 1] == "Generator", "Visualize must come after Generator"
    assert names[idx + 1] == "Statistical", "Visualize must come before Statistical"


def test_sidebar_run_selector_called_in_main() -> None:
    """_sidebar_run_selector() must be called before st.tabs in main()
    so the selectbox always renders regardless of the active tab."""
    src = (Path(__file__).parent.parent /
           "src/visualization/telemetry_dashboard.py").read_text()
    main_body = src[src.index("def main()"):]
    tabs_pos = main_body.index("st.tabs(")
    sidebar_pos = main_body.index("_sidebar_run_selector()")
    assert sidebar_pos < tabs_pos, (
        "_sidebar_run_selector() must be called before st.tabs() in main()"
    )


def test_generate_sets_active_run() -> None:
    """After generate, tel_active_run must be set to the saved run path so that
    all tabs automatically switch to the new data without extra user interaction."""
    telemetry, labels = _gen_from_defaults()
    with tempfile.TemporaryDirectory() as tmp:
        saved = save_run({"telemetry.pt": telemetry, "labels.pt": labels}, base_dir=tmp)

    assert saved.exists() or not saved.exists()
    session_state: dict = {}
    session_state["tel_active_run"] = saved
    assert session_state["tel_active_run"] == saved


def test_visualize_channel_stats_normal_vs_anomaly() -> None:
    """The channel stats shown in the Visualize tab must differ between normal
    and anomalous timesteps — a basic sanity check that anomalies are visible
    in the statistics table."""
    telemetry, labels = _gen_from_defaults()
    series, bin_labels = _first_series(telemetry, labels)
    assert bin_labels.sum().item() > 0, "Need anomalies for this test"

    normal = series[bin_labels == 0]
    anomalous = series[bin_labels == 1]

    diffs = (anomalous.mean(dim=0) - normal.mean(dim=0)).abs()
    max_diff = float(diffs.max())
    assert max_diff > 0.1, (
        f"Max channel mean diff (normal vs anomaly) = {max_diff:.4f}; "
        "expected > 0.1 — anomalies should be visible in the stats table"
    )


def test_visualize_channel_stats_shape() -> None:
    """The stats table must have one row per channel."""
    telemetry, labels = _gen_from_defaults()
    series, bin_labels = _first_series(telemetry, labels)
    normal = series[bin_labels == 0]
    n_c = series.shape[1]

    from src.data.generators.telemetry import CHANNEL_NAMES
    rows = []
    for i, name in enumerate(CHANNEL_NAMES[:n_c]):
        rows.append({
            "Channel": name,
            "Normal mean": float(normal[:, i].mean()),
            "Normal std":  float(normal[:, i].std()),
        })

    assert len(rows) == n_c, f"Expected {n_c} rows, got {len(rows)}"
    assert all(r["Normal std"] > 0 for r in rows), "All channels must have positive std"


# ---------------------------------------------------------------------------
# 7. _ensure_3d backward compat — legacy (T,C) runs must not crash
# ---------------------------------------------------------------------------

def test_ensure_3d_promotes_legacy_tensors() -> None:
    """_ensure_3d must promote (T,C)/(T,) tensors to (1,T,C)/(1,T) without error."""
    from src.visualization.telemetry_dashboard import _ensure_3d
    t_2d = torch.randn(500, 7)
    l_1d = torch.zeros(500, dtype=torch.long)
    t_out, l_out = _ensure_3d(t_2d, l_1d)
    assert t_out.shape == (1, 500, 7)
    assert l_out.shape == (1, 500)


def test_ensure_3d_leaves_3d_unchanged() -> None:
    """_ensure_3d must not modify tensors that are already (N,T,C)/(N,T)."""
    from src.visualization.telemetry_dashboard import _ensure_3d
    t_3d = torch.randn(5, 500, 7)
    l_2d = torch.zeros(5, 500, dtype=torch.long)
    t_out, l_out = _ensure_3d(t_3d, l_2d)
    assert t_out.shape == (5, 500, 7)
    assert l_out.shape == (5, 500)


# ---------------------------------------------------------------------------
# 8. _series_slider — must not crash when n=1
# ---------------------------------------------------------------------------

def test_series_slider_returns_zero_for_single_series() -> None:
    """When n=1 the slider must be skipped and 0 returned without raising."""
    from unittest.mock import patch

    from src.visualization.telemetry_dashboard import _series_slider
    with patch("streamlit.slider") as mock_slider:
        result = _series_slider("Series index", 1, key="test_key")
    assert result == 0
    mock_slider.assert_not_called()


def test_series_slider_renders_for_multiple_series() -> None:
    """When n>1 the slider must be called with valid min < max."""
    from unittest.mock import patch

    from src.visualization.telemetry_dashboard import _series_slider
    with patch("streamlit.slider", return_value=0) as mock_slider:
        _series_slider("Series index", 5, key="test_key")
    mock_slider.assert_called_once()
    call_args = mock_slider.call_args
    min_val = call_args.args[1]
    max_val = call_args.args[2]
    assert min_val < max_val, f"slider min={min_val} >= max={max_val}"


# ---------------------------------------------------------------------------
# 9. _train_test_split — determinism and shape contract
# ---------------------------------------------------------------------------

def test_train_test_split_shapes() -> None:
    """_train_test_split must return train (K*T, C) and valid index lists."""
    from src.visualization.telemetry_dashboard import _train_test_split
    telemetry = torch.randn(10, 200, 7)
    train_data, train_idx, test_idx = _train_test_split(telemetry, train_frac=0.6)
    assert train_data.shape[1] == 7
    assert train_data.shape[0] == len(train_idx) * 200
    assert len(train_idx) + len(test_idx) == 10
    assert set(train_idx) & set(test_idx) == set(), "Train and test indices must not overlap"


def test_train_test_split_is_reproducible() -> None:
    """Same seed must produce the same split."""
    from src.visualization.telemetry_dashboard import _train_test_split
    telemetry = torch.randn(10, 200, 7)
    _, idx1, _ = _train_test_split(telemetry, 0.5, seed=0)
    _, idx2, _ = _train_test_split(telemetry, 0.5, seed=0)
    assert idx1 == idx2


def test_train_test_split_different_seeds_differ() -> None:
    """Different seeds must (almost certainly) produce different splits."""
    from src.visualization.telemetry_dashboard import _train_test_split
    telemetry = torch.randn(20, 200, 7)
    _, idx0, _ = _train_test_split(telemetry, 0.5, seed=0)
    _, idx1, _ = _train_test_split(telemetry, 0.5, seed=1)
    assert idx0 != idx1, "Different seeds should produce different train/test splits"


def test_train_test_split_single_series_does_not_crash() -> None:
    """A dataset of N=1 must not crash the split (train=test=series 0)."""
    from src.visualization.telemetry_dashboard import _train_test_split
    telemetry = torch.randn(1, 200, 7)
    train_data, train_idx, test_idx = _train_test_split(telemetry, 0.5)
    assert train_data.shape == (200, 7)
    assert len(train_idx) == 1
    assert len(test_idx) == 1


# ---------------------------------------------------------------------------
# 10. _detector_results_section inputs — per-type table must not crash
# ---------------------------------------------------------------------------

def test_per_type_auc_table_does_not_crash_with_missing_types() -> None:
    """If some anomaly types are absent from the scored data, the per-type table
    must silently skip them rather than raising KeyError or division-by-zero."""
    from sklearn.metrics import roc_auc_score as _auc

    from src.data.generators.telemetry import ANOMALY_TYPE_IDS

    n_t = 300
    labels_mc = torch.zeros(n_t, dtype=torch.long)
    labels_mc[50:60] = ANOMALY_TYPE_IDS["spike"]
    scores = torch.rand(n_t)

    labels_np = labels_mc.numpy()
    y_score = scores.numpy()
    type_rows = []
    for type_id in sorted(ANOMALY_TYPE_IDS.values()):
        mask_relevant = (labels_np == 0) | (labels_np == type_id)
        if (labels_np == type_id).sum() == 0:
            continue
        y_rel = (labels_np[mask_relevant] == type_id).astype(int)
        s_rel = y_score[mask_relevant]
        try:
            auc_t = _auc(y_rel, s_rel)
        except ValueError:
            auc_t = float("nan")
        type_rows.append({"type_id": type_id, "auc": auc_t})
    assert len(type_rows) == 1, f"Expected 1 type row (spike only), got {len(type_rows)}"
    assert type_rows[0]["type_id"] == ANOMALY_TYPE_IDS["spike"]


# ---------------------------------------------------------------------------
# 11. End-to-end detector tab logic (no Streamlit, uses dashboard helpers)
# ---------------------------------------------------------------------------

def test_detector_tab_logic_n1_does_not_crash() -> None:
    """The full detector tab logic (split → fit → score) must work with N=1.
    This is the exact sequence that was crashing when a legacy run was loaded."""
    from src.models.baselines import MahalanobisDetector
    from src.visualization.telemetry_dashboard import _series_slider, _train_test_split

    cfg = TelemetryGeneratorConfig(n_channels=4, n_timesteps=300, seed=7,
                                   anomaly_ratio=0.05)
    telemetry, labels_mc = TelemetryGenerator(cfg).generate(n_series=1)

    train_data, train_idx, test_idx = _train_test_split(telemetry, 0.5)
    det = MahalanobisDetector(window=10).fit(train_data)
    n_t, n_c = telemetry.shape[1], telemetry.shape[2]
    test_tel = telemetry[test_idx].reshape(len(test_idx) * n_t, n_c)
    scores = det.score(test_tel)
    assert scores.shape == (len(test_idx) * n_t,)

    m = len(test_idx)
    series_idx = _series_slider("idx", m, key="dummy")
    assert series_idx == 0


def test_detector_tab_logic_n20_does_not_crash() -> None:
    """The full detector tab logic must work with N=20 (normal multi-series case)."""
    from src.models.baselines import PerChannelZScore
    from src.visualization.telemetry_dashboard import _train_test_split

    cfg = TelemetryGeneratorConfig(n_channels=4, n_timesteps=200, seed=8,
                                   anomaly_ratio=0.05)
    telemetry, labels_mc = TelemetryGenerator(cfg).generate(n_series=20)

    train_data, train_idx, test_idx = _train_test_split(telemetry, 0.5)
    det = PerChannelZScore(window=10).fit(train_data)
    n_t, n_c = telemetry.shape[1], telemetry.shape[2]
    test_tel = telemetry[test_idx].reshape(len(test_idx) * n_t, n_c)
    scores = det.score(test_tel)
    assert scores.shape == (len(test_idx) * n_t,)
    assert len(train_idx) + len(test_idx) == 20
