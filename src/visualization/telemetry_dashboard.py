from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve

from src.data.generators.telemetry import (
    ANOMALY_TYPE_IDS,
    ANOMALY_TYPES,
    CHANNEL_NAMES,
    TelemetryGenerator,
    TelemetryGeneratorConfig,
)
from src.models.baselines import CUSUMDetector, MahalanobisDetector, PerChannelZScore
from src.models.classical import IsolationForestDetector, OneClassSVMDetector
from src.models.deep import LSTMAutoencoderDetector
from src.utils.metrics import best_f1_threshold as _best_f1_threshold
from src.visualization.data_io import (
    list_runs,
    load_tensor,
    save_run,
)

st.set_page_config(page_title="Telemetry Anomaly Sandbox", layout="wide")

DEFAULT_DATA_DIR = "data/synthetic/telemetry"
_FILENAMES = ("telemetry.pt", "labels.pt")

GEN_DEFAULTS: dict = {
    "tel_n_series": 20,
    "tel_n_channels": 7,
    "tel_n_timesteps": 1000,
    "tel_noise_std": 0.05,
    "tel_orbital_period": 200,
    "tel_anomaly_ratio": 0.05,
    "tel_seed": 42,
}

# Color map for anomaly types in plots (type ID → matplotlib color)
_TYPE_COLORS: dict[int, str] = {
    ANOMALY_TYPE_IDS["spike"]:             "red",
    ANOMALY_TYPE_IDS["step"]:              "orange",
    ANOMALY_TYPE_IDS["ramp"]:              "gold",
    ANOMALY_TYPE_IDS["correlation_break"]: "purple",
}
_TYPE_NAMES: dict[int, str] = {v: k for k, v in ANOMALY_TYPE_IDS.items()}


# ---------------------------------------------------------------------------
# Sidebar — shared run selector
# ---------------------------------------------------------------------------

def _sidebar_run_selector() -> None:
    """Sidebar widget: pick the active dataset used by all tabs.

    Stores the selected run Path in ``st.session_state["tel_active_run"]``.
    When the Generator tab saves a new run it sets this key directly so that
    all other tabs immediately switch to the fresh data.
    """
    with st.sidebar:
        st.header("Active Dataset")
        runs = list_runs(DEFAULT_DATA_DIR, _FILENAMES)
        if not runs:
            st.info(
                f"No saved runs found in `{DEFAULT_DATA_DIR}`. "
                "Use the **Generator** tab to create one."
            )
            st.session_state["tel_active_run"] = None
            return

        options = [r.name for r in runs]
        run_map = {r.name: r for r in runs}

        active = st.session_state.get("tel_active_run")
        if isinstance(active, Path):
            active_name = active.name
        elif isinstance(active, str):
            active_name = Path(active).name
        else:
            active_name = None
        default_idx = options.index(active_name) if active_name in options else 0

        selected = st.selectbox(
            "Run",
            options,
            index=default_idx,
            key="tel_active_run_select",
            help="Newest first. All tabs use this run.",
        )
        st.session_state["tel_active_run"] = run_map[selected]
        st.caption(f"`{run_map[selected]}`")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _series_slider(label: str, n: int, key: str) -> int:
    """Return a series index slider, or 0 silently when there is only one series."""
    if n <= 1:
        return 0
    return st.slider(label, 0, n - 1, 0, key=key)


def _ensure_3d(
    telemetry: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Upgrade legacy (T, C) / (T,) tensors to (1, T, C) / (1, T).

    Old runs were saved in the pre-N-series format. Rather than crashing, we
    silently promote them so the dashboard still works; a banner tells the user
    to regenerate for full multi-series functionality.
    """
    if telemetry.dim() == 2:
        st.warning(
            "This run was saved in the old single-series format `(T, C)`. "
            "It will be treated as a dataset of N=1 series. "
            "**Click Generate to create a fresh multi-series run.**"
        )
        telemetry = telemetry.unsqueeze(0)
        labels = labels.unsqueeze(0)
    return telemetry, labels


def _load_data() -> tuple[torch.Tensor, torch.Tensor] | None:
    """Load the active run selected in the sidebar.

    Returns (telemetry (N,T,C), labels (N,T) multi-class), or None when no
    run has been selected yet.
    """
    active = st.session_state.get("tel_active_run")
    if active is None:
        return None
    run_path = Path(active) if not isinstance(active, Path) else active
    if not run_path.exists():
        st.warning(f"Run `{run_path}` no longer exists — select another in the sidebar.")
        return None
    tensors = tuple(load_tensor(run_path / f) for f in _FILENAMES)
    return _ensure_3d(tensors[0], tensors[1])


def _plot_timeseries(
    telemetry: torch.Tensor,
    labels_mc: torch.Tensor,
    scores: torch.Tensor | None = None,
    title: str = "Telemetry",
    channel_names: list[str] | None = None,
) -> None:
    """Plot a single time series (T, C) with anomaly-type color coding.

    *labels_mc* is a 1-D tensor of shape (T,) with values 0=normal or a
    positive anomaly-type ID from ANOMALY_TYPE_IDS.
    """
    n_t, n_c = telemetry.shape
    t = np.arange(n_t)
    labels_np = labels_mc.numpy()
    n_rows = n_c + (1 if scores is not None else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 1.8 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]
    names = channel_names or [f"ch{i}" for i in range(n_c)]

    # Build per-type anomaly spans for color-coded background shading
    type_spans: dict[int, list[tuple[int, int]]] = {tid: [] for tid in _TYPE_COLORS}
    for tid in _TYPE_COLORS:
        mask = (labels_np == tid).astype(int)
        starts = np.where(np.diff(mask, prepend=0) == 1)[0]
        ends = np.where(np.diff(mask, append=0) == -1)[0]
        type_spans[tid] = list(zip(starts.tolist(), ends.tolist()))

    for i in range(n_c):
        ax = axes[i]
        ax.plot(t, telemetry[:, i].numpy(), lw=0.8)
        for tid, spans in type_spans.items():
            color = _TYPE_COLORS[tid]
            for j, (s, e) in enumerate(spans):
                lbl = _TYPE_NAMES[tid] if i == 0 and j == 0 else ""
                ax.axvspan(s, e, alpha=0.25, color=color, label=lbl)
        ax.set_ylabel(names[i], fontsize=8)
        ax.tick_params(labelsize=7)
    if scores is not None:
        ax = axes[n_c]
        ax.plot(t, scores.numpy(), lw=0.8, color="steelblue", label="score")
        for tid, spans in type_spans.items():
            for s, e in spans:
                ax.axvspan(s, e, alpha=0.2, color=_TYPE_COLORS[tid])
        ax.set_ylabel("score", fontsize=8)
        ax.tick_params(labelsize=7)
    if any(len(v) > 0 for v in type_spans.values()):
        axes[0].legend(fontsize=7, loc="upper right")
    axes[0].set_title(title)
    axes[-1].set_xlabel("Timestep")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def _train_test_split(
    telemetry: torch.Tensor,
    train_frac: float,
    seed: int = 0,
) -> tuple[torch.Tensor, list[int], list[int]]:
    """Random series-level train/test split — no labels used.

    Returns:
        train_data  : (K*T, C) concatenated training series
        train_idx   : indices of training series
        test_idx    : indices of test series
    """
    n = telemetry.shape[0]
    k = max(1, int(n * train_frac))
    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()
    train_idx = perm[:k]
    test_idx = perm[k:] if k < n else perm[:1]
    n_t, n_c = telemetry.shape[1], telemetry.shape[2]
    train_data = telemetry[train_idx].reshape(len(train_idx) * n_t, n_c)
    return train_data, train_idx, test_idx


def _detector_tab(
    tab_key: str,
    header: str,
    description: str,
    detector_factories: dict,
    window_range: tuple[int, int, int],
    default_index: int = 0,
) -> None:
    """Generic skeleton shared by the Statistical and ML detector tabs.

    *detector_factories* maps display name → callable(window) → unfitted detector.
    Training uses a label-free random series-level split.
    Data is loaded from the run selected in the sidebar.
    """
    st.header(header)
    st.markdown(description)

    result = _load_data()
    if result is None:
        st.info("Select a run in the **sidebar** to get started.")
        return
    telemetry, labels_mc = result

    n = telemetry.shape[0]
    train_frac = st.slider(
        "Training series fraction", 0.1, 0.9, 0.5, 0.05, key=f"{tab_key}_frac",
        help="Fraction of series used for training. No labels are used — split is random.",
    )
    detector_name = st.selectbox(
        "Detector",
        list(detector_factories),
        index=default_index,
        key=f"{tab_key}_det",
    )
    w_min, w_max, w_default = window_range
    window = st.slider("Window size", w_min, w_max, w_default, key=f"{tab_key}_window")

    if st.button("Run", key=f"{tab_key}_run"):
        train_data, train_idx, test_idx = _train_test_split(telemetry, train_frac)
        det = detector_factories[detector_name](window).fit(train_data)
        n_t, n_c = telemetry.shape[1], telemetry.shape[2]
        test_tel = telemetry[test_idx].reshape(len(test_idx) * n_t, n_c)
        test_labels_mc = labels_mc[test_idx].reshape(-1)
        scores = det.score(test_tel)
        st.session_state[f"{tab_key}_scores"] = scores
        st.session_state[f"{tab_key}_labels_mc"] = test_labels_mc
        st.session_state[f"{tab_key}_test_tel"] = telemetry[test_idx]
        st.session_state[f"{tab_key}_test_labels_mc"] = labels_mc[test_idx]
        st.session_state.pop(f"{tab_key}_threshold", None)
        st.caption(
            f"Trained on {len(train_idx)}/{n} series, scored on {len(test_idx)}/{n} series."
        )

    if f"{tab_key}_scores" not in st.session_state:
        st.info("Click **Run** to fit and score.")
        return
    _detector_results_section(
        tab_key,
        st.session_state[f"{tab_key}_scores"],
        st.session_state[f"{tab_key}_labels_mc"],
        st.session_state[f"{tab_key}_test_tel"],
        st.session_state[f"{tab_key}_test_labels_mc"],
    )


def _detector_results_section(
    tab_key: str,
    scores: torch.Tensor,
    labels_mc: torch.Tensor,
    test_tel: torch.Tensor,
    test_labels_mc: torch.Tensor,
) -> None:
    """Show ROC curve, score distribution, per-type metrics, and time series.

    Args:
        scores        : (M*T,) anomaly scores for all test timesteps
        labels_mc     : (M*T,) multi-class labels for all test timesteps
        test_tel      : (M, T, C) test series for time series display
        test_labels_mc: (M, T) per-series multi-class labels for display
    """
    y_true_bin = (labels_mc > 0).long().numpy()
    y_score = scores.numpy()

    auc = roc_auc_score(y_true_bin, y_score)
    st.metric("ROC AUC (all anomaly types)", f"{auc:.3f}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Score Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(y_score[y_true_bin == 0], bins=40, alpha=0.6, label="Normal", density=True)
        ax.hist(y_score[y_true_bin == 1], bins=40, alpha=0.6, label="Anomaly", density=True)
        ax.set_xlabel("Anomaly score")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    best_thresh = float(np.clip(
        _best_f1_threshold(y_true_bin, y_score), y_score.min(), y_score.max()
    ))
    thresh_key = f"{tab_key}_threshold"
    if thresh_key not in st.session_state:
        st.session_state[thresh_key] = best_thresh

    def _reset_threshold(key=thresh_key, val=best_thresh):
        st.session_state[key] = val

    col_slider, col_btn = st.columns([5, 1])
    with col_btn:
        st.button(
            "Reset to best-F1",
            on_click=_reset_threshold,
            key=f"{tab_key}_thresh_reset",
            help=f"Reset threshold to the F1-optimal value ({best_thresh:.4f})",
        )
    with col_slider:
        threshold = st.slider(
            "Decision threshold",
            min_value=float(y_score.min()),
            max_value=float(y_score.max()),
            step=(float(y_score.max()) - float(y_score.min())) / 100,
            key=thresh_key,
        )
    preds = (y_score >= threshold).astype(int)
    c1, c2, c3 = st.columns(3)
    c1.metric("Precision", f"{precision_score(y_true_bin, preds, zero_division=0):.3f}")
    c2.metric("Recall", f"{recall_score(y_true_bin, preds, zero_division=0):.3f}")
    c3.metric("F1", f"{f1_score(y_true_bin, preds, zero_division=0):.3f}")

    # Per-anomaly-type AUC breakdown
    st.subheader("Per-Type Detection (AUC)")
    labels_mc_np = labels_mc.numpy()
    type_rows = []
    for type_id in sorted(_TYPE_NAMES):
        mask_relevant = (labels_mc_np == 0) | (labels_mc_np == type_id)
        if mask_relevant.sum() == 0 or (labels_mc_np == type_id).sum() == 0:
            continue
        y_rel = (labels_mc_np[mask_relevant] == type_id).astype(int)
        s_rel = y_score[mask_relevant]
        try:
            auc_t = roc_auc_score(y_rel, s_rel)
        except ValueError:
            auc_t = float("nan")
        type_rows.append({
            "Type": _TYPE_NAMES[type_id],
            "Anomalous timesteps": int((labels_mc_np == type_id).sum()),
            "AUC (type vs normal)": f"{auc_t:.3f}",
        })
    if type_rows:
        st.table(type_rows)

    st.subheader("Time Series with Anomaly Overlay")
    m = test_tel.shape[0]
    series_idx = _series_slider("Test series index", m, key=f"{tab_key}_series_idx")
    n_t = test_tel.shape[1]
    series_scores = scores[series_idx * n_t : (series_idx + 1) * n_t]
    _plot_timeseries(
        test_tel[series_idx],
        test_labels_mc[series_idx],
        scores=series_scores,
        title=f"Test series {series_idx} + detector score",
        channel_names=CHANNEL_NAMES[:test_tel.shape[2]],
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

def tab_generator() -> None:
    st.header("Generator")
    st.markdown("Experiment with synthetic telemetry parameters and generate a dataset.")

    for key, val in GEN_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val

    col1, col2 = st.columns([1, 3])
    with col1:
        n_series = st.slider("Number of series", 5, 100, 20, 5, key="tel_n_series")
        n_channels = st.slider("Channels", 2, 10, 7, key="tel_n_channels")
        n_timesteps = st.slider(
            "Timesteps per series", 200, 5000, 1000, 100, key="tel_n_timesteps"
        )
        noise_std = st.slider("Noise std", 0.01, 0.5, 0.05, 0.01, key="tel_noise_std")
        orbital_period = st.slider(
            "Orbital period (steps)", 50, 500, 200, 10, key="tel_orbital_period"
        )
        anomaly_ratio = st.slider(
            "Anomaly ratio", 0.01, 0.3, 0.05, 0.01, key="tel_anomaly_ratio"
        )
        anomaly_types = st.multiselect(
            "Anomaly types",
            options=list(ANOMALY_TYPES),
            default=list(ANOMALY_TYPES),
            key="tel_anomaly_types",
        )
        seed = st.number_input("Seed", min_value=0, value=42, step=1, key="tel_seed")

        def _reset_to_defaults() -> None:
            for k, v in GEN_DEFAULTS.items():
                st.session_state[k] = v

        col_gen, col_reset = st.columns(2)
        with col_reset:
            st.button("Reset to defaults", key="tel_reset", on_click=_reset_to_defaults)
        with col_gen:
            generate_clicked = st.button("Generate", key="tel_generate")

        if generate_clicked:
            cfg = TelemetryGeneratorConfig(
                n_channels=n_channels,
                n_timesteps=n_timesteps,
                noise_std=noise_std,
                orbital_period_steps=orbital_period,
                anomaly_ratio=anomaly_ratio,
                anomaly_types=anomaly_types or list(ANOMALY_TYPES),
                seed=seed,
            )
            telemetry, labels_mc = TelemetryGenerator(cfg).generate(n_series=n_series)
            st.session_state["tel_telemetry"] = telemetry
            st.session_state["tel_labels_mc"] = labels_mc
            saved = save_run(
                {"telemetry.pt": telemetry, "labels.pt": labels_mc},
                base_dir=DEFAULT_DATA_DIR,
            )
            st.session_state["tel_active_run"] = saved
            st.success(
                f"Saved to `{saved}` — sidebar updated, all tabs now use this run."
            )

    with col2:
        if "tel_telemetry" in st.session_state:
            telemetry = st.session_state["tel_telemetry"]
            labels_mc = st.session_state["tel_labels_mc"]
            n = telemetry.shape[0]
            series_idx = _series_slider("Preview series", n, key="tel_gen_series_idx")
            _plot_timeseries(
                telemetry[series_idx],
                labels_mc[series_idx],
                title=f"Generated series {series_idx} / {n}",
                channel_names=CHANNEL_NAMES[:telemetry.shape[2]],
            )
        else:
            st.info("Click **Generate** to create the dataset.")


def tab_visualize() -> None:
    st.header("Visualize")
    st.markdown("Inspect the active dataset. Select a different run in the sidebar.")

    result = _load_data()
    if result is None:
        st.info("Select a run in the **sidebar** to get started.")
        return
    telemetry, labels_mc = result

    n, n_t, n_c = telemetry.shape
    n_anomaly = int((labels_mc > 0).sum().item())
    total_steps = n * n_t
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Series", n)
    c2.metric("Timesteps / series", n_t)
    c3.metric("Anomalous timesteps (total)", n_anomaly)
    c4.metric("Anomaly ratio", f"{n_anomaly / total_steps:.1%}")

    type_counts = {
        _TYPE_NAMES[tid]: int((labels_mc == tid).sum().item())
        for tid in sorted(_TYPE_NAMES)
        if (labels_mc == tid).any()
    }
    if type_counts:
        st.markdown(
            "**Anomaly type breakdown:** "
            + "  |  ".join(f"{k}: {v}" for k, v in type_counts.items())
        )

    series_idx = _series_slider("Series index", n, key="tel_viz_series_idx")
    series = telemetry[series_idx]
    series_labels = labels_mc[series_idx]
    names = CHANNEL_NAMES[:n_c]

    st.subheader(f"Series {series_idx} — Time Series")
    _plot_timeseries(series, series_labels, title="", channel_names=names)

    st.subheader("Channel Statistics (this series)")
    bin_labels = (series_labels > 0)
    normal = series[~bin_labels]
    anom = series[bin_labels] if bin_labels.any() else None
    rows = []
    for i, name in enumerate(names):
        row: dict = {
            "Channel": name,
            "Normal mean": f"{float(normal[:, i].mean()):.4f}",
            "Normal std":  f"{float(normal[:, i].std()):.4f}",
            "Normal min":  f"{float(normal[:, i].min()):.4f}",
            "Normal max":  f"{float(normal[:, i].max()):.4f}",
        }
        if anom is not None and len(anom) > 0:
            row["Anomaly mean"] = f"{float(anom[:, i].mean()):.4f}"
            row["Anomaly std"]  = f"{float(anom[:, i].std()):.4f}"
        rows.append(row)
    st.table(rows)


_STAT_DETECTORS = {
    "PerChannelZScore": lambda w: PerChannelZScore(window=w),
    "Mahalanobis":      lambda w: MahalanobisDetector(window=w),
    "CUSUM":            lambda _w: CUSUMDetector(),
}

_ML_DETECTORS = {
    "Isolation Forest": lambda w: IsolationForestDetector(window=w),
    "One-Class SVM":    lambda w: OneClassSVMDetector(window=w),
}


def tab_statistical() -> None:
    _detector_tab(
        tab_key="tel_stat",
        header="Statistical Detectors",
        description=(
            "Three classical baselines: per-channel z-score, multivariate Mahalanobis "
            "distance, and CUSUM drift detection."
        ),
        detector_factories=_STAT_DETECTORS,
        window_range=(5, 100, 20),
        default_index=1,
    )


def tab_ml() -> None:
    _detector_tab(
        tab_key="tel_ml",
        header="ML Detectors",
        description="Isolation Forest and One-Class SVM operating on sliding windows.",
        detector_factories=_ML_DETECTORS,
        window_range=(5, 50, 10),
        default_index=1,
    )


def tab_deep() -> None:
    st.header("LSTM Autoencoder")
    st.markdown(
        "Trains an LSTM encoder-decoder on normal windows. "
        "Anomaly score = per-timestep reconstruction MSE."
    )

    result = _load_data()
    if result is None:
        st.info("Select a run in the **sidebar** to get started.")
        return
    telemetry, labels_mc = result

    n = telemetry.shape[0]
    train_frac = st.slider(
        "Training series fraction", 0.1, 0.9, 0.5, 0.05, key="tel_deep_frac",
        help="Fraction of series used for training. No labels are used — split is random.",
    )

    col1, col2, col3 = st.columns(3)
    window = col1.slider("Window", 10, 100, 30, key="tel_deep_window")
    hidden = col2.slider("Hidden size", 8, 128, 32, key="tel_deep_hidden")
    epochs = col3.slider("Epochs", 5, 100, 20, key="tel_deep_epochs")

    if st.button("Train & Score", key="tel_deep_run"):
        train_data, train_idx, test_idx = _train_test_split(telemetry, train_frac)
        with st.spinner("Training LSTM autoencoder…"):
            det = LSTMAutoencoderDetector(
                window=window, hidden_size=hidden, n_epochs=epochs
            ).fit(train_data)
        n_t, n_c = telemetry.shape[1], telemetry.shape[2]
        test_tel = telemetry[test_idx].reshape(len(test_idx) * n_t, n_c)
        test_labels_mc = labels_mc[test_idx].reshape(-1)
        scores = det.score(test_tel)
        st.session_state["tel_deep_scores"] = scores
        st.session_state["tel_deep_labels_mc"] = test_labels_mc
        st.session_state["tel_deep_test_tel"] = telemetry[test_idx]
        st.session_state["tel_deep_test_labels_mc"] = labels_mc[test_idx]
        st.session_state["tel_deep_losses"] = det.train_losses
        st.session_state.pop("tel_deep_threshold", None)
        st.caption(
            f"Trained on {len(train_idx)}/{n} series, scored on {len(test_idx)}/{n} series."
        )

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(det.train_losses)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE loss")
        ax.set_title("Training loss")
        st.pyplot(fig)
        plt.close()

    if "tel_deep_scores" not in st.session_state:
        st.info("Click **Train & Score** to fit the autoencoder.")
        return

    if "tel_deep_losses" in st.session_state:
        with st.expander("Training loss curve"):
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(st.session_state["tel_deep_losses"])
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE loss")
            st.pyplot(fig)
            plt.close()

    _detector_results_section(
        "tel_deep",
        st.session_state["tel_deep_scores"],
        st.session_state["tel_deep_labels_mc"],
        st.session_state["tel_deep_test_tel"],
        st.session_state["tel_deep_test_labels_mc"],
    )


def tab_comparison() -> None:
    st.header("Model Comparison")
    st.markdown(
        "Runs all detectors with default settings and overlays their ROC curves. "
        "Uses a random 50/50 series-level split — no labels used for training."
    )

    result = _load_data()
    if result is None:
        st.info("Select a run in the **sidebar** to get started.")
        return
    telemetry, labels_mc = result

    if not st.button("Run All Detectors", key="tel_cmp_run"):
        st.info("Click **Run All Detectors** to compare.")
        return

    train_data, _, test_idx = _train_test_split(telemetry, 0.5)
    n_t, n_c = telemetry.shape[1], telemetry.shape[2]
    test_tel = telemetry[test_idx].reshape(len(test_idx) * n_t, n_c)
    y_mc = labels_mc[test_idx].reshape(-1).numpy()
    y_true = (y_mc > 0).astype(int)

    detectors: dict = {
        "Z-Score": PerChannelZScore(window=20),
        "Mahalanobis": MahalanobisDetector(window=20),
        "CUSUM": CUSUMDetector(),
        "Isolation Forest": IsolationForestDetector(window=20),
        "One-Class SVM": OneClassSVMDetector(window=20),
    }

    results: dict[str, dict] = {}
    progress = st.progress(0.0)
    for i, (name, det) in enumerate(detectors.items()):
        with st.spinner(f"Fitting {name}…"):
            det.fit(train_data)
            scores = det.score(test_tel).numpy()
            auc = roc_auc_score(y_true, scores)
            best_f1 = 0.0
            for pct in range(1, 100):
                thresh = float(np.percentile(scores, pct))
                preds = (scores >= thresh).astype(int)
                f1 = f1_score(y_true, preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
            fpr, tpr, _ = roc_curve(y_true, scores)
            results[name] = {"auc": auc, "f1": best_f1, "fpr": fpr, "tpr": tpr}
        progress.progress((i + 1) / (len(detectors) + 1))

    with st.spinner("Fitting LSTM autoencoder (may take a moment)…"):
        lstm = LSTMAutoencoderDetector(window=30, hidden_size=32, n_epochs=20).fit(train_data)
        scores = lstm.score(test_tel).numpy()
        auc = roc_auc_score(y_true, scores)
        best_f1 = 0.0
        for pct in range(1, 100):
            thresh = float(np.percentile(scores, pct))
            preds = (scores >= thresh).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
        fpr, tpr, _ = roc_curve(y_true, scores)
        results["LSTM Autoencoder"] = {"auc": auc, "f1": best_f1, "fpr": fpr, "tpr": tpr}
    progress.progress(1.0)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROC Curves")
        fig, ax = plt.subplots(figsize=(6, 5))
        for name, r in results.items():
            ax.plot(r["fpr"], r["tpr"], label=f"{name} (AUC={r['auc']:.2f})")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Metrics Table")
        rows = [
            {"Detector": name, "AUC": f"{r['auc']:.3f}", "Best F1": f"{r['f1']:.3f}"}
            for name, r in results.items()
        ]
        st.table(rows)


def main() -> None:
    _sidebar_run_selector()
    st.title("Telemetry Anomaly Sandbox")
    tabs = st.tabs(["Generator", "Visualize", "Statistical", "ML", "Deep", "Comparison"])
    with tabs[0]:
        tab_generator()
    with tabs[1]:
        tab_visualize()
    with tabs[2]:
        tab_statistical()
    with tabs[3]:
        tab_ml()
    with tabs[4]:
        tab_deep()
    with tabs[5]:
        tab_comparison()


if __name__ == "__main__":
    main()
