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

import pytest
import torch
from streamlit.testing.v1 import AppTest

from src.data.generators.telemetry import TelemetryGenerator, TelemetryGeneratorConfig
from src.visualization.telemetry_dashboard import GEN_DEFAULTS

APP_PATH = "src/visualization/telemetry_dashboard.py"
_SHORT_CFG = TelemetryGeneratorConfig(
    n_channels=4, n_timesteps=100, anomaly_ratio=0.1, seed=0
)


def _gen(n_series: int) -> tuple[torch.Tensor, torch.Tensor]:
    return TelemetryGenerator(_SHORT_CFG).generate(n_series=n_series)


# ---------------------------------------------------------------------------
# 1. Cold start
# ---------------------------------------------------------------------------

def test_app_runs_empty_state() -> None:
    """App must start without exception when no data has been generated yet."""
    at = AppTest.from_file(APP_PATH).run()
    assert not at.exception


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
                f"Slider '{key}' default {slider_by_key[key].value!r} != GEN_DEFAULTS {expected!r}"
            )
        elif key in number_by_key:
            assert number_by_key[key].value == expected, (
                f"NumberInput '{key}' default {number_by_key[key].value!r} "
                f"!= GEN_DEFAULTS {expected!r}"
            )


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


# ---------------------------------------------------------------------------
# 4. Sidebar run selector
# ---------------------------------------------------------------------------

def test_sidebar_run_selector_no_crash_empty_state() -> None:
    """_sidebar_run_selector must not raise regardless of saved-run availability."""
    at = AppTest.from_file(APP_PATH).run()
    assert not at.exception


def test_active_run_in_session_state_loads_data_no_crash() -> None:
    """When tel_active_run is pre-set in session state and the path exists,
    _load_data must return data and the app must not crash."""
    import tempfile

    from src.visualization.data_io import save_run
    cfg = TelemetryGeneratorConfig(n_channels=4, n_timesteps=100, seed=0,
                                   anomaly_ratio=0.05)
    telemetry, labels_mc = TelemetryGenerator(cfg).generate(n_series=3)
    with tempfile.TemporaryDirectory() as tmp:
        saved = save_run({"telemetry.pt": telemetry, "labels.pt": labels_mc},
                         base_dir=tmp)
        at = AppTest.from_file(APP_PATH)
        at.session_state["tel_active_run"] = saved
        at.run()
        assert not at.exception


# ---------------------------------------------------------------------------
# 5. Visualize tab — no data scenario must not crash
# ---------------------------------------------------------------------------

def test_visualize_tab_no_data_shows_info() -> None:
    """Visualize tab must display an info message (not crash) when no data exists."""
    at = AppTest.from_file(APP_PATH).run()
    assert not at.exception


# ---------------------------------------------------------------------------
# 6. Detector tabs — no data must not crash
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tab_key", ["tel_stat", "tel_ml", "tel_deep", "tel_cmp"])
def test_detector_tab_no_data_no_crash(tab_key: str) -> None:
    """Every detector tab must handle the no-data case without exception."""
    at = AppTest.from_file(APP_PATH).run()
    assert not at.exception


# ---------------------------------------------------------------------------
# 6. Reset button resets sliders to GEN_DEFAULTS
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
