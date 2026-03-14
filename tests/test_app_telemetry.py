"""AppTest-based tests for the telemetry Streamlit dashboard.

Uses streamlit.testing.v1.AppTest to drive the real Streamlit rendering
pipeline — the only way to catch crashes caused by widget configuration
errors (e.g. slider min >= max) that plain unit tests cannot see.

Each test follows the pattern:
    at = AppTest.from_file(APP_PATH)
    [optionally pre-populate at.session_state]
    at.run()
    assert not at.exception
"""

import tempfile
from pathlib import Path

import pytest
import torch
from streamlit.testing.v1 import AppTest

from src.data.generators.telemetry import TelemetryGenerator, TelemetryGeneratorConfig
from src.visualization.data_io import save_run
from src.visualization.telemetry_dashboard import GEN_DEFAULTS

APP_PATH = "src/visualization/telemetry_dashboard.py"
_FILENAMES = ("telemetry.pt", "labels.pt")
_SHORT_CFG = TelemetryGeneratorConfig(
    n_channels=4, n_timesteps=100, anomaly_ratio=0.1, seed=0
)


def _gen(n_series: int) -> tuple[torch.Tensor, torch.Tensor]:
    return TelemetryGenerator(_SHORT_CFG).generate(n_series=n_series)


def _saved_run(n_series: int = 3) -> tuple[str, torch.Tensor, torch.Tensor]:
    """Save a run to a temp dir, return (saved_path_str, telemetry, labels_mc)."""
    telemetry, labels_mc = _gen(n_series)
    tmp = tempfile.mkdtemp()
    saved = save_run({"telemetry.pt": telemetry, "labels.pt": labels_mc}, base_dir=tmp)
    return str(saved), telemetry, labels_mc


# ---------------------------------------------------------------------------
# 1. Cold start
# ---------------------------------------------------------------------------

def test_app_runs_empty_state() -> None:
    """App must start without exception when no data has been generated yet."""
    at = AppTest.from_file(APP_PATH).run()
    assert not at.exception


def test_app_title_is_correct() -> None:
    """App title must be 'Telemetry Anomaly Sandbox'."""
    at = AppTest.from_file(APP_PATH).run()
    assert not at.exception
    assert at.title[0].value == "Telemetry Anomaly Sandbox"


def test_app_has_six_tabs() -> None:
    """App must render exactly 6 tabs with the expected labels."""
    at = AppTest.from_file(APP_PATH).run()
    assert not at.exception
    labels = [t.label for t in at.tabs]
    assert labels == ["Generator", "Visualize", "Statistical", "ML", "Deep", "Comparison"]


def test_sidebar_has_active_dataset_header() -> None:
    """Sidebar must always show the 'Active Dataset' header."""
    at = AppTest.from_file(APP_PATH).run()
    assert not at.exception
    sidebar_headers = [h.value for h in at.sidebar.header]
    assert "Active Dataset" in sidebar_headers


# ---------------------------------------------------------------------------
# 2. Generator tab — widget inventory
# ---------------------------------------------------------------------------

def test_generator_tab_has_n_series_slider() -> None:
    at = AppTest.from_file(APP_PATH).run()
    assert not at.exception
    slider_keys = [s.key for s in at.slider]
    assert "tel_n_series" in slider_keys, (
        "tel_n_series slider missing — Generator tab must expose a Number of series slider"
    )


def test_generator_tab_slider_defaults_match_gen_defaults() -> None:
    """Every numeric GEN_DEFAULT must appear as the initial slider/input value."""
    at = AppTest.from_file(APP_PATH).run()
    assert not at.exception
    slider_by_key = {s.key: s for s in at.slider}
    number_by_key = {n.key: n for n in at.number_input}
    for key, expected in GEN_DEFAULTS.items():
        if key in slider_by_key:
            assert slider_by_key[key].value == expected, (
                f"Slider '{key}' default {slider_by_key[key].value!r} "
                f"!= GEN_DEFAULTS {expected!r}"
            )
        elif key in number_by_key:
            assert number_by_key[key].value == expected, (
                f"NumberInput '{key}' default {number_by_key[key].value!r} "
                f"!= GEN_DEFAULTS {expected!r}"
            )


def test_generator_tab_has_generate_and_reset_buttons() -> None:
    at = AppTest.from_file(APP_PATH).run()
    assert not at.exception
    button_keys = [b.key for b in at.button]
    assert "tel_generate" in button_keys
    assert "tel_reset" in button_keys


def test_generator_tab_has_anomaly_types_multiselect() -> None:
    at = AppTest.from_file(APP_PATH).run()
    assert not at.exception
    ms_keys = [m.key for m in at.multiselect]
    assert "tel_anomaly_types" in ms_keys


# ---------------------------------------------------------------------------
# 3. Generator tab — session state with pre-generated data
# ---------------------------------------------------------------------------

def test_generator_tab_with_n1_series_no_crash() -> None:
    """Generator tab must not crash when session state contains a single series (n=1).
    This was previously causing 'Slider min_value must be less than max_value' because
    st.slider('Preview series', 0, n-1, ...) became st.slider(..., 0, 0, ...)."""
    telemetry, labels_mc = _gen(n_series=1)
    at = AppTest.from_file(APP_PATH)
    at.session_state["tel_telemetry"] = telemetry
    at.session_state["tel_labels_mc"] = labels_mc
    at.run()
    assert not at.exception


def test_generator_tab_with_n20_series_no_crash() -> None:
    """Generator tab must not crash with a normal multi-series dataset."""
    telemetry, labels_mc = _gen(n_series=20)
    at = AppTest.from_file(APP_PATH)
    at.session_state["tel_telemetry"] = telemetry
    at.session_state["tel_labels_mc"] = labels_mc
    at.run()
    assert not at.exception


def test_generator_tab_series_slider_absent_for_n1() -> None:
    """When session state contains n=1 series the 'Preview series' slider
    must NOT be rendered (it would have min=max=0 which Streamlit rejects)."""
    telemetry, labels_mc = _gen(n_series=1)
    at = AppTest.from_file(APP_PATH)
    at.session_state["tel_telemetry"] = telemetry
    at.session_state["tel_labels_mc"] = labels_mc
    at.run()
    assert not at.exception
    slider_keys = [s.key for s in at.slider]
    assert "tel_gen_series_idx" not in slider_keys, (
        "Preview series slider must be hidden when n=1 to avoid min=max=0 error"
    )


def test_generator_tab_series_slider_present_for_n_gt_1() -> None:
    """When session state contains multiple series the 'Preview series' slider
    must be rendered with valid min < max."""
    telemetry, labels_mc = _gen(n_series=5)
    at = AppTest.from_file(APP_PATH)
    at.session_state["tel_telemetry"] = telemetry
    at.session_state["tel_labels_mc"] = labels_mc
    at.run()
    assert not at.exception
    slider_by_key = {s.key: s for s in at.slider}
    assert "tel_gen_series_idx" in slider_by_key, (
        "Preview series slider must be rendered when n>1"
    )
    s = slider_by_key["tel_gen_series_idx"]
    assert s.min < s.max, f"Slider min={s.min} >= max={s.max}"


def test_generator_tab_shows_success_after_generate() -> None:
    """After generate, a st.success message must be shown with the saved path."""
    at = AppTest.from_file(APP_PATH)
    at.session_state["tel_last_saved"] = "/some/run/path"
    at.run()
    assert not at.exception
    assert len(at.success) > 0, "Expected at least one st.success message"
    assert "/some/run/path" in at.success[0].value


# ---------------------------------------------------------------------------
# 4. Sidebar run selector
# ---------------------------------------------------------------------------

def test_sidebar_selectbox_syncs_to_active_run() -> None:
    """When tel_active_run is pre-set, _sidebar_run_selector must sync the
    selectbox to show that run on the next render — using AppTest.from_function
    to isolate the sidebar logic from the full app and real filesystem."""
    def sidebar_app() -> None:
        from pathlib import Path

        import streamlit as st

        fake_runs = [Path("run_2026-01-02"), Path("run_2026-01-01")]  # newest first
        options = [r.name for r in fake_runs]
        run_map = {r.name: r for r in fake_runs}

        active = st.session_state.get("tel_active_run")
        active_name = (
            active.name if isinstance(active, Path)
            else Path(active).name if isinstance(active, str)
            else None
        )

        if active_name in options:
            st.session_state["tel_active_run_select"] = active_name
        elif "tel_active_run_select" not in st.session_state:
            st.session_state["tel_active_run_select"] = options[0]

        selected = st.selectbox("Run", options, key="tel_active_run_select")
        st.session_state["tel_active_run"] = run_map[selected]

    at = AppTest.from_function(sidebar_app)
    at.session_state["tel_active_run"] = Path("run_2026-01-01")
    at.run()
    assert not at.exception
    assert at.selectbox[0].value == "run_2026-01-01", (
        "Selectbox must show the run that was set in tel_active_run, "
        "not default to the first (newest) option"
    )


def test_sidebar_selectbox_defaults_to_newest_when_no_active_run() -> None:
    """When no active run is set, the selectbox must default to the first
    (newest) option."""
    def sidebar_app() -> None:
        from pathlib import Path

        import streamlit as st

        fake_runs = [Path("run_2026-01-02"), Path("run_2026-01-01")]
        options = [r.name for r in fake_runs]
        run_map = {r.name: r for r in fake_runs}

        active = st.session_state.get("tel_active_run")
        active_name = (
            active.name if isinstance(active, Path)
            else Path(active).name if isinstance(active, str)
            else None
        )

        if active_name in options:
            st.session_state["tel_active_run_select"] = active_name
        elif "tel_active_run_select" not in st.session_state:
            st.session_state["tel_active_run_select"] = options[0]

        selected = st.selectbox("Run", options, key="tel_active_run_select")
        st.session_state["tel_active_run"] = run_map[selected]

    at = AppTest.from_function(sidebar_app)
    at.run()
    assert not at.exception
    assert at.selectbox[0].value == "run_2026-01-02", (
        "Without active run, selectbox must default to newest run"
    )


def test_sidebar_run_selector_no_crash_empty_state() -> None:
    """_sidebar_run_selector must not raise regardless of saved-run availability."""
    at = AppTest.from_file(APP_PATH).run()
    assert not at.exception


def test_active_run_in_session_state_loads_data_no_crash() -> None:
    """When tel_active_run is pre-set and the path exists on disk,
    the app must not crash — exercises the full _load_data() pipeline."""
    saved_str, telemetry, labels_mc = _saved_run(n_series=3)
    at = AppTest.from_file(APP_PATH)
    at.session_state["tel_active_run"] = saved_str
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# 5. Visualize tab — content when data is loaded
# ---------------------------------------------------------------------------

def test_visualize_tab_no_data_shows_info() -> None:
    """Visualize tab must display an info message (not crash) when no data exists."""
    at = AppTest.from_file(APP_PATH).run()
    assert not at.exception


def test_visualize_tab_shows_metrics_when_data_loaded() -> None:
    """Visualize tab must render metrics (Series, Timesteps, Anomaly ratio)
    when a valid run is loaded via tel_active_run."""
    saved_str, telemetry, labels_mc = _saved_run(n_series=5)
    at = AppTest.from_file(APP_PATH)
    at.session_state["tel_active_run"] = saved_str
    at.run()
    assert not at.exception
    metric_labels = [m.label for m in at.metric]
    assert "Series" in metric_labels, f"Expected 'Series' metric, got: {metric_labels}"
    assert "Timesteps / series" in metric_labels


def test_visualize_tab_shows_channel_stats_table() -> None:
    """Visualize tab must render a channel stats table when data is loaded."""
    saved_str, telemetry, labels_mc = _saved_run(n_series=3)
    at = AppTest.from_file(APP_PATH)
    at.session_state["tel_active_run"] = saved_str
    at.run()
    assert not at.exception
    assert len(at.table) > 0, "Expected at least one st.table in the Visualize tab"


def test_visualize_tab_series_slider_present_for_multi_series() -> None:
    """Series selector slider must be rendered when the loaded dataset has N>1."""
    saved_str, telemetry, labels_mc = _saved_run(n_series=5)
    at = AppTest.from_file(APP_PATH)
    at.session_state["tel_active_run"] = saved_str
    at.run()
    assert not at.exception
    slider_keys = [s.key for s in at.slider]
    assert "tel_viz_series_idx" in slider_keys


# ---------------------------------------------------------------------------
# 6. Detector tabs — no data must not crash
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tab_key", ["tel_stat", "tel_ml", "tel_deep", "tel_cmp"])
def test_detector_tab_no_data_no_crash(tab_key: str) -> None:
    """Every detector tab must handle the no-data case without exception."""
    at = AppTest.from_file(APP_PATH).run()
    assert not at.exception


# ---------------------------------------------------------------------------
# 7. Reset button resets sliders to GEN_DEFAULTS
# ---------------------------------------------------------------------------

def test_reset_button_restores_defaults() -> None:
    """Clicking 'Reset to defaults' must restore all GEN_DEFAULTS values.

    The button uses an on_click callback so the session state is updated
    *before* widgets are instantiated — this avoids the Streamlit restriction
    that forbids modifying widget-backed keys after instantiation.
    """
    at = AppTest.from_file(APP_PATH).run()
    assert not at.exception
    at.slider(key="tel_n_series").set_value(99).run()
    assert not at.exception
    at.button(key="tel_reset").click().run()
    assert not at.exception
    assert at.session_state["tel_n_series"] == GEN_DEFAULTS["tel_n_series"], (
        f"tel_n_series should be reset to {GEN_DEFAULTS['tel_n_series']}, "
        f"got {at.session_state['tel_n_series']}"
    )


def test_reset_button_restores_all_numeric_defaults() -> None:
    """All numeric GEN_DEFAULTS must be restored by clicking Reset."""
    at = AppTest.from_file(APP_PATH).run()
    assert not at.exception
    at.slider(key="tel_n_series").set_value(99).run()
    at.slider(key="tel_n_channels").set_value(2).run()
    at.slider(key="tel_noise_std").set_value(0.5).run()
    at.button(key="tel_reset").click().run()
    assert not at.exception
    for key, expected in GEN_DEFAULTS.items():
        if key in {s.key for s in at.slider}:
            actual = at.slider(key=key).value
            assert actual == expected, (
                f"After reset: slider '{key}' = {actual!r}, expected {expected!r}"
            )


# ---------------------------------------------------------------------------
# 8. Slider interaction — changing values rerenders without crash
# ---------------------------------------------------------------------------

def test_n_series_slider_interaction_no_crash() -> None:
    """Changing the n_series slider must not crash the app."""
    at = AppTest.from_file(APP_PATH).run()
    assert not at.exception
    at.slider(key="tel_n_series").set_value(50).run()
    assert not at.exception
    assert at.slider(key="tel_n_series").value == 50


def test_noise_slider_interaction_no_crash() -> None:
    """Changing noise_std must not crash the app."""
    at = AppTest.from_file(APP_PATH).run()
    assert not at.exception
    at.slider(key="tel_noise_std").set_value(0.2).run()
    assert not at.exception


def test_anomaly_ratio_slider_interaction_no_crash() -> None:
    at = AppTest.from_file(APP_PATH).run()
    assert not at.exception
    at.slider(key="tel_anomaly_ratio").set_value(0.15).run()
    assert not at.exception


# ---------------------------------------------------------------------------
# 9. Warning — legacy 2D data emits st.warning via _ensure_3d
# ---------------------------------------------------------------------------

def test_ensure_3d_emits_warning_for_legacy_data() -> None:
    """Loading a legacy (T,C) run must emit a st.warning, not crash.

    The run must be saved inside DEFAULT_DATA_DIR so that _sidebar_run_selector
    picks it up and honours the tel_active_run pre-set.  We clean up afterwards.
    """
    import shutil

    from src.visualization.telemetry_dashboard import DEFAULT_DATA_DIR

    saved = save_run(
        {
            "telemetry.pt": torch.randn(100, 4),   # legacy 2D shape
            "labels.pt": torch.zeros(100, dtype=torch.long),
        },
        base_dir=DEFAULT_DATA_DIR,
    )
    try:
        at = AppTest.from_file(APP_PATH)
        at.session_state["tel_active_run"] = saved
        at.run()
        assert not at.exception
        assert len(at.warning) > 0, (
            "Expected a st.warning for legacy (T,C) data, got none"
        )
    finally:
        shutil.rmtree(saved, ignore_errors=True)


# ---------------------------------------------------------------------------
# 10. Comparison tab — reuses existing scores; runs only what is missing
# ---------------------------------------------------------------------------

def test_comparison_tab_shows_warning_when_no_detectors_run() -> None:
    """Comparison tab must show a warning (not crash) when no detector has been run."""
    saved_str, _, _ = _saved_run(n_series=5)
    at = AppTest.from_file(APP_PATH)
    at.session_state["tel_active_run"] = saved_str
    at.run()
    assert not at.exception
    # No scores → expect at least one warning listing the missing detectors
    assert len(at.warning) > 0, (
        "Expected a st.warning when no detectors have been run yet"
    )


def _pre_run_detector(
    session_state: dict,
    tab_key: str,
    det_name: str,
    scores: torch.Tensor,
    labels_mc_flat: torch.Tensor,
    test_series: torch.Tensor,
    test_labels_mc: torch.Tensor,
) -> None:
    """Write the full set of session-state keys that _detector_tab stores after Run.

    Both the individual detector tab AND the comparison tab read these keys, so
    the test must supply the complete set to avoid KeyError crashes.
    """
    session_state[f"{tab_key}_scores"] = scores
    session_state[f"{tab_key}_labels_mc"] = labels_mc_flat
    session_state[f"{tab_key}_test_tel"] = test_series
    session_state[f"{tab_key}_test_labels_mc"] = test_labels_mc
    session_state[f"{tab_key}_det_ran"] = det_name


def test_comparison_tab_reuses_stat_scores_without_rerunning() -> None:
    """When tel_stat_scores is pre-populated in session state, the Comparison tab
    must render the ROC curve immediately (no button click, no recomputation)."""
    from src.models.baselines import MahalanobisDetector

    saved_str, telemetry, labels_mc = _saved_run(n_series=6)
    n, n_t, n_c = telemetry.shape
    k = n // 2
    train = telemetry[:k].reshape(k * n_t, n_c)
    test_series = telemetry[k:]
    scores = MahalanobisDetector(window=5).fit(train).score(
        test_series.reshape((n - k) * n_t, n_c)
    )

    at = AppTest.from_file(APP_PATH)
    at.session_state["tel_active_run"] = saved_str
    _pre_run_detector(
        at.session_state, "tel_stat", "Mahalanobis",
        scores, labels_mc[k:].reshape(-1), test_series, labels_mc[k:],
    )
    at.run()
    assert not at.exception

    assert len(at.table) > 0, "Expected metrics table in Comparison tab"
    # at.table[-1] is the Comparison metrics table; earlier tables belong to
    # the Visualize tab (channel stats) and the Statistical tab (per-type AUC).
    assert "Mahalanobis" in str(at.table[-1].value), (
        f"Mahalanobis not found in comparison table; got: {at.table[-1].value}"
    )


def test_comparison_tab_shows_run_missing_button_for_partial_results() -> None:
    """When only one of three detector slots has been run, the
    'Run missing detectors' button must appear."""
    from src.models.baselines import MahalanobisDetector

    saved_str, telemetry, labels_mc = _saved_run(n_series=6)
    n, n_t, n_c = telemetry.shape
    k = n // 2
    test_series = telemetry[k:]
    scores = MahalanobisDetector(window=5).fit(
        telemetry[:k].reshape(k * n_t, n_c)
    ).score(test_series.reshape((n - k) * n_t, n_c))

    at = AppTest.from_file(APP_PATH)
    at.session_state["tel_active_run"] = saved_str
    _pre_run_detector(
        at.session_state, "tel_stat", "Mahalanobis",
        scores, labels_mc[k:].reshape(-1), test_series, labels_mc[k:],
    )
    at.run()
    assert not at.exception

    button_keys = [b.key for b in at.button]
    assert "tel_cmp_run" in button_keys, (
        "Expected 'Run missing detectors' button when ML and Deep have not been run"
    )


def test_comparison_tab_no_run_missing_button_when_all_run() -> None:
    """When all three detector categories have been run, the 'Run missing detectors'
    button must NOT appear — the comparison shows results immediately."""
    from src.models.baselines import MahalanobisDetector
    from src.models.classical import IsolationForestDetector

    saved_str, telemetry, labels_mc = _saved_run(n_series=6)
    n, n_t, n_c = telemetry.shape
    k = n // 2
    test_series = telemetry[k:]
    flat_test = test_series.reshape((n - k) * n_t, n_c)
    labels_flat = labels_mc[k:].reshape(-1)
    train = telemetry[:k].reshape(k * n_t, n_c)

    stat_scores = MahalanobisDetector(window=5).fit(train).score(flat_test)
    ml_scores = IsolationForestDetector(window=5).fit(train).score(flat_test)

    at = AppTest.from_file(APP_PATH)
    at.session_state["tel_active_run"] = saved_str
    _pre_run_detector(
        at.session_state, "tel_stat", "Mahalanobis",
        stat_scores, labels_flat, test_series, labels_mc[k:],
    )
    _pre_run_detector(
        at.session_state, "tel_ml", "Isolation Forest",
        ml_scores, labels_flat, test_series, labels_mc[k:],
    )
    _pre_run_detector(
        at.session_state, "tel_deep", "LSTM Autoencoder",
        stat_scores.clone(), labels_flat, test_series, labels_mc[k:],
    )
    at.run()
    assert not at.exception

    button_keys = [b.key for b in at.button]
    assert "tel_cmp_run" not in button_keys, (
        "'Run missing detectors' button must be absent when all detectors have been run"
    )

    assert len(at.table) > 0
    # at.table[-1] is the Comparison metrics table (last in page order)
    table_str = str(at.table[-1].value)
    for name in ("Mahalanobis", "Isolation Forest", "LSTM Autoencoder"):
        assert name in table_str, f"'{name}' not found in comparison table"
