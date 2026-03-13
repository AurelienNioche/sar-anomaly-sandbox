from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve

from src.data.generators.telemetry import (
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
    "tel_n_channels": 7,
    "tel_n_timesteps": 1000,
    "tel_noise_std": 0.05,
    "tel_orbital_period": 200,
    "tel_anomaly_ratio": 0.05,
    "tel_seed": 42,
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _run_selectbox(tab_key: str) -> Path | None:
    """Render a selectbox of available runs; return the selected run Path or None."""
    runs = list_runs(DEFAULT_DATA_DIR, _FILENAMES)
    if not runs:
        st.info(f"No saved runs found in `{DEFAULT_DATA_DIR}` — generate data first.")
        return None
    options = [r.name for r in runs]
    selected = st.selectbox(
        "Saved run",
        options,
        index=0,
        key=f"{tab_key}_run_select",
        help="Runs are listed newest first.",
    )
    return runs[options.index(selected)]


def _load_data(tab_key: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    run_path = _run_selectbox(tab_key)
    if run_path is None:
        return None
    st.caption(f"Loaded `{run_path}`")
    tensors = tuple(load_tensor(run_path / f) for f in _FILENAMES)
    return tensors[0], tensors[1]


def _plot_timeseries(
    telemetry: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor | None = None,
    title: str = "Telemetry",
    channel_names: list[str] | None = None,
) -> None:
    n_t, n_c = telemetry.shape
    t = np.arange(n_t)
    anom_mask = labels.numpy().astype(bool)
    n_rows = n_c + (1 if scores is not None else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 1.8 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]
    names = channel_names or [f"ch{i}" for i in range(n_c)]
    anom_starts = np.where(np.diff(anom_mask.astype(int), prepend=0) == 1)[0]
    anom_ends   = np.where(np.diff(anom_mask.astype(int), append=0) == -1)[0]
    for i in range(n_c):
        ax = axes[i]
        ax.plot(t, telemetry[:, i].numpy(), lw=0.8, label=names[i])
        for s, e in zip(anom_starts, anom_ends):
            lbl = "anomaly" if s == anom_starts[0] else ""
            ax.axvspan(s, e, alpha=0.25, color="red", label=lbl)
        ax.set_ylabel(names[i], fontsize=8)
        ax.tick_params(labelsize=7)
    if scores is not None:
        ax = axes[n_c]
        ax.plot(t, scores.numpy(), lw=0.8, color="orange", label="score")
        for s, e in zip(anom_starts, anom_ends):
            ax.axvspan(s, e, alpha=0.25, color="red")
        ax.set_ylabel("score", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_title(title)
    axes[-1].set_xlabel("Timestep")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def _split_normal(
    telemetry: torch.Tensor,
    labels: torch.Tensor,
    frac: float,
) -> torch.Tensor:
    """Return the training-normal slice: first *frac* of anomaly-free timesteps."""
    normal = telemetry[labels == 0]
    return normal[:max(1, int(len(normal) * frac))]


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
    """
    st.header(header)
    st.markdown(description)

    result = _load_data(tab_key)
    if result is None:
        st.info("Select a data source above.")
        return
    telemetry, labels = result

    train_frac = st.slider(
        "Training fraction (normal)", 0.1, 0.9, 0.5, 0.05, key=f"{tab_key}_frac"
    )
    train = _split_normal(telemetry, labels, train_frac)
    detector_name = st.selectbox(
        "Detector",
        list(detector_factories),
        index=default_index,
        key=f"{tab_key}_det",
    )
    w_min, w_max, w_default = window_range
    window = st.slider("Window size", w_min, w_max, w_default, key=f"{tab_key}_window")

    if st.button("Run", key=f"{tab_key}_run"):
        det = detector_factories[detector_name](window).fit(train)
        scores = det.score(telemetry)
        st.session_state[f"{tab_key}_scores"] = scores
        st.session_state[f"{tab_key}_labels"] = labels
        st.session_state[f"{tab_key}_telemetry"] = telemetry
        st.session_state.pop(f"{tab_key}_threshold", None)

    if f"{tab_key}_scores" not in st.session_state:
        st.info("Click **Run** to fit and score.")
        return
    _detector_results_section(
        tab_key,
        st.session_state[f"{tab_key}_scores"],
        st.session_state[f"{tab_key}_labels"],
        st.session_state[f"{tab_key}_telemetry"],
    )


def _detector_results_section(
    tab_key: str,
    scores: torch.Tensor,
    labels: torch.Tensor,
    telemetry: torch.Tensor,
) -> None:
    y_true = labels.numpy()
    y_score = scores.numpy()

    auc = roc_auc_score(y_true, y_score)
    st.metric("ROC AUC", f"{auc:.3f}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_true, y_score)
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
        ax.hist(y_score[y_true == 0], bins=40, alpha=0.6, label="Normal", density=True)
        ax.hist(y_score[y_true == 1], bins=40, alpha=0.6, label="Anomaly", density=True)
        ax.set_xlabel("Anomaly score")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    best_thresh = float(np.clip(
        _best_f1_threshold(y_true, y_score), y_score.min(), y_score.max()
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
    c1.metric("Precision", f"{precision_score(y_true, preds, zero_division=0):.3f}")
    c2.metric("Recall", f"{recall_score(y_true, preds, zero_division=0):.3f}")
    c3.metric("F1", f"{f1_score(y_true, preds, zero_division=0):.3f}")

    st.subheader("Time Series with Anomaly Overlay")
    _plot_timeseries(
        telemetry, labels, scores=scores,
        title="Telemetry + detector score",
        channel_names=CHANNEL_NAMES[:telemetry.shape[1]],
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

def tab_generator() -> None:
    st.header("Generator")
    st.markdown("Experiment with synthetic telemetry parameters and generate data.")

    for key, val in GEN_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val

    col1, col2 = st.columns([1, 3])
    with col1:
        n_channels = st.slider("Channels", 2, 10, 7, key="tel_n_channels")
        n_timesteps = st.slider("Timesteps", 200, 5000, 1000, 100, key="tel_n_timesteps")
        noise_std = st.slider("Noise std", 0.01, 0.5, 0.05, 0.01, key="tel_noise_std")
        orbital_period = st.slider(
            "Orbital period (steps)", 50, 500, 200, 10, key="tel_orbital_period"
        )
        anomaly_ratio = st.slider("Anomaly ratio", 0.01, 0.3, 0.05, 0.01, key="tel_anomaly_ratio")
        anomaly_types = st.multiselect(
            "Anomaly types",
            options=list(ANOMALY_TYPES),
            default=list(ANOMALY_TYPES),
            key="tel_anomaly_types",
        )
        seed = st.number_input("Seed", min_value=0, value=42, step=1, key="tel_seed")

        col_gen, col_reset = st.columns(2)
        with col_reset:
            if st.button("Reset to defaults", key="tel_reset"):
                for k, v in GEN_DEFAULTS.items():
                    st.session_state[k] = v
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
            telemetry, labels = TelemetryGenerator(cfg).generate()
            st.session_state["tel_telemetry"] = telemetry
            st.session_state["tel_labels"] = labels
            saved = save_run(
                {"telemetry.pt": telemetry, "labels.pt": labels},
                base_dir=DEFAULT_DATA_DIR,
            )
            saved_str = str(saved)
            st.session_state["tel_saved_path"] = saved_str
            for suffix in ("stat", "ml", "deep", "cmp"):
                st.session_state[f"tel_{suffix}_dir"] = saved_str

        if st.session_state.get("tel_saved_path"):
            st.success(
                f"Saved to `{st.session_state['tel_saved_path']}` — "
                "all detector tabs updated automatically."
            )

    with col2:
        if "tel_telemetry" in st.session_state:
            _plot_timeseries(
                st.session_state["tel_telemetry"],
                st.session_state["tel_labels"],
                title="Generated telemetry",
                channel_names=CHANNEL_NAMES[:st.session_state["tel_telemetry"].shape[1]],
            )
        else:
            st.info("Click **Generate** to create telemetry with the current parameters.")


def tab_visualize() -> None:
    st.header("Visualize")
    st.markdown(
        "Browse any saved run and inspect the raw telemetry. "
        "The run open here automatically becomes the data source for all detector tabs."
    )

    run_path = _run_selectbox("tel_viz")
    telemetry: torch.Tensor | None = None
    labels: torch.Tensor | None = None

    if run_path is not None:
        resolved = str(run_path)
        st.caption(f"Loaded `{resolved}`")
        tensors = tuple(load_tensor(run_path / f) for f in _FILENAMES)
        telemetry, labels = tensors[0], tensors[1]
        if st.session_state.get("tel_viz_last_synced") != resolved:
            for suffix in ("stat", "ml", "deep", "cmp"):
                st.session_state[f"tel_{suffix}_dir"] = resolved
            st.session_state["tel_viz_last_synced"] = resolved

    if telemetry is None or labels is None:
        st.info("No data loaded — generate data in the **Generator** tab first.")
        return

    n_t, n_c = telemetry.shape
    n_anomaly = int(labels.sum().item())
    c1, c2, c3 = st.columns(3)
    c1.metric("Timesteps", n_t)
    c2.metric("Anomalous timesteps", n_anomaly)
    c3.metric("Anomaly ratio", f"{n_anomaly / n_t:.1%}")

    st.subheader("Time Series")
    names = CHANNEL_NAMES[:n_c]
    _plot_timeseries(telemetry, labels, title="", channel_names=names)

    st.subheader("Channel Statistics")
    normal = telemetry[labels == 0]
    anom = telemetry[labels == 1] if n_anomaly > 0 else None
    rows = []
    for i, name in enumerate(names):
        row: dict = {
            "Channel": name,
            "Normal mean": f"{float(normal[:, i].mean()):.4f}",
            "Normal std":  f"{float(normal[:, i].std()):.4f}",
            "Normal min":  f"{float(normal[:, i].min()):.4f}",
            "Normal max":  f"{float(normal[:, i].max()):.4f}",
        }
        if anom is not None:
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

    result = _load_data("tel_deep")
    if result is None:
        st.info("Select a data source above.")
        return
    telemetry, labels = result

    train_frac = st.slider("Training fraction (normal)", 0.1, 0.9, 0.5, 0.05, key="tel_deep_frac")
    train = _split_normal(telemetry, labels, train_frac)

    col1, col2, col3 = st.columns(3)
    window = col1.slider("Window", 10, 100, 30, key="tel_deep_window")
    hidden = col2.slider("Hidden size", 8, 128, 32, key="tel_deep_hidden")
    epochs = col3.slider("Epochs", 5, 100, 20, key="tel_deep_epochs")

    if st.button("Train & Score", key="tel_deep_run"):
        with st.spinner("Training LSTM autoencoder…"):
            det = LSTMAutoencoderDetector(
                window=window, hidden_size=hidden, n_epochs=epochs
            ).fit(train)
        scores = det.score(telemetry)
        st.session_state["tel_deep_scores"] = scores
        st.session_state["tel_deep_labels"] = labels
        st.session_state["tel_deep_telemetry"] = telemetry
        st.session_state["tel_deep_losses"] = det.train_losses
        st.session_state.pop("tel_deep_threshold", None)

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
        st.session_state["tel_deep_labels"],
        st.session_state["tel_deep_telemetry"],
    )


def tab_comparison() -> None:
    st.header("Model Comparison")
    st.markdown(
        "Runs all detectors with default settings and overlays their ROC curves. "
        "Uses a fixed 50/50 train/test split on normal data."
    )

    result = _load_data("tel_cmp")
    if result is None:
        st.info("Select a data source above.")
        return
    telemetry, labels = result

    if not st.button("Run All Detectors", key="tel_cmp_run"):
        st.info("Click **Run All Detectors** to compare.")
        return

    train = _split_normal(telemetry, labels, 0.5)

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
            det.fit(train)
            scores = det.score(telemetry).numpy()
            y_true = labels.numpy()
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
        progress.progress((i + 1) / len(detectors))

    with st.spinner("Fitting LSTM autoencoder (may take a moment)…"):
        lstm = LSTMAutoencoderDetector(window=30, hidden_size=32, n_epochs=20).fit(train)
        scores = lstm.score(telemetry).numpy()
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
