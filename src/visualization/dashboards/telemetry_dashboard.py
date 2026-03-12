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
from src.visualization.dashboards.data_io import (
    load_tensors_from_dir,
    load_tensors_from_upload,
    save_run,
)

st.set_page_config(page_title="Telemetry Anomaly Sandbox", layout="wide")

DEFAULT_DATA_DIR = "data/telemetry"
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

def _load_data(tab_key: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    dir_path = st.text_input(
        "Data directory",
        value=DEFAULT_DATA_DIR,
        key=f"{tab_key}_dir",
        help="Path containing telemetry.pt and labels.pt (or a parent with "
             "timestamped run sub-folders — the latest run is auto-selected).",
    )
    result_tensors = None
    if dir_path:
        raw = load_tensors_from_dir(dir_path, _FILENAMES)
        if raw is not None:
            (telemetry, labels), resolved = raw
            result_tensors = (telemetry, labels)
            resolved_str = str(resolved)
            msg = (
                f"Auto-selected latest run: `{resolved_str}`"
                if resolved_str != dir_path
                else f"Loaded from `{resolved_str}`"
            )
            st.success(msg)
        else:
            st.warning(f"`{dir_path}` not found or contains no valid run data.")

    uploaded = st.file_uploader(
        "Or drag-and-drop a folder to override",
        accept_multiple_files="directory",
        type=None,
        key=f"{tab_key}_upload",
    )
    if uploaded:
        tensors = load_tensors_from_upload(uploaded, _FILENAMES)
        if tensors is None:
            st.error("Expected telemetry.pt and labels.pt in the selected folder.")
        else:
            result_tensors = (tensors[0], tensors[1])
    return result_tensors


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
    for i in range(n_c):
        ax = axes[i]
        ax.plot(t, telemetry[:, i].numpy(), lw=0.8, label=names[i])
        ax.fill_between(t, ax.get_ylim()[0], ax.get_ylim()[1],
                        where=anom_mask, alpha=0.25, color="red", label="anomaly")
        ax.set_ylabel(names[i], fontsize=8)
        ax.tick_params(labelsize=7)
    if scores is not None:
        ax = axes[n_c]
        ax.plot(t, scores.numpy(), lw=0.8, color="orange", label="score")
        ax.fill_between(t, 0, scores.numpy().max(),
                        where=anom_mask, alpha=0.25, color="red")
        ax.set_ylabel("score", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_title(title)
    axes[-1].set_xlabel("Timestep")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def _best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    _, _, thresholds = roc_curve(y_true, y_score)
    best_thresh, best_f1 = float(thresholds[0]), 0.0
    for t in thresholds:
        preds = (y_score >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)
    return best_thresh


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

    default_thresh = st.session_state.get(
        f"{tab_key}_threshold", _best_f1_threshold(y_true, y_score)
    )
    threshold = st.slider(
        "Decision threshold",
        min_value=float(y_score.min()),
        max_value=float(y_score.max()),
        value=float(np.clip(default_thresh, y_score.min(), y_score.max())),
        step=(float(y_score.max()) - float(y_score.min())) / 100,
        key=f"{tab_key}_threshold",
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
            st.session_state["tel_saved_path"] = None

        if "tel_telemetry" in st.session_state:
            if st.button("Save to disk", key="tel_save"):
                saved = save_run(
                    {"telemetry.pt": st.session_state["tel_telemetry"],
                     "labels.pt": st.session_state["tel_labels"]},
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


def tab_statistical() -> None:
    st.header("Statistical Detectors")
    st.markdown(
        "Three classical baselines: per-channel z-score, multivariate Mahalanobis distance, "
        "and CUSUM drift detection."
    )

    result = _load_data("tel_stat")
    if result is None:
        st.info("Select a data source above.")
        return
    telemetry, labels = result

    train_frac = st.slider("Training fraction (normal)", 0.1, 0.9, 0.5, 0.05, key="tel_stat_frac")
    normal = telemetry[labels == 0]
    n_train = max(1, int(len(normal) * train_frac))
    train = normal[:n_train]
    detector_name = st.selectbox(
        "Detector",
        ["PerChannelZScore", "Mahalanobis", "CUSUM"],
        key="tel_stat_det",
    )
    window = st.slider("Window size", 5, 100, 20, key="tel_stat_window")

    if st.button("Run", key="tel_stat_run"):
        if detector_name == "PerChannelZScore":
            det = PerChannelZScore(window=window).fit(train)
        elif detector_name == "Mahalanobis":
            det = MahalanobisDetector(window=window).fit(train)
        else:
            det = CUSUMDetector().fit(train)
        scores = det.score(telemetry)
        st.session_state["tel_stat_scores"] = scores
        st.session_state["tel_stat_labels"] = labels
        st.session_state["tel_stat_telemetry"] = telemetry
        st.session_state.pop("tel_stat_threshold", None)

    if "tel_stat_scores" not in st.session_state:
        st.info("Click **Run** to fit and score.")
        return
    _detector_results_section(
        "tel_stat",
        st.session_state["tel_stat_scores"],
        st.session_state["tel_stat_labels"],
        st.session_state["tel_stat_telemetry"],
    )


def tab_ml() -> None:
    st.header("ML Detectors")
    st.markdown("Isolation Forest and One-Class SVM operating on sliding windows.")

    result = _load_data("tel_ml")
    if result is None:
        st.info("Select a data source above.")
        return
    telemetry, labels = result

    train_frac = st.slider("Training fraction (normal)", 0.1, 0.9, 0.5, 0.05, key="tel_ml_frac")
    normal = telemetry[labels == 0]
    n_train = max(1, int(len(normal) * train_frac))
    train = normal[:n_train]
    detector_name = st.selectbox(
        "Detector", ["Isolation Forest", "One-Class SVM"], key="tel_ml_det"
    )
    window = st.slider("Window size", 5, 50, 20, key="tel_ml_window")

    if st.button("Run", key="tel_ml_run"):
        if detector_name == "Isolation Forest":
            det = IsolationForestDetector(window=window).fit(train)
        else:
            det = OneClassSVMDetector(window=window).fit(train)
        scores = det.score(telemetry)
        st.session_state["tel_ml_scores"] = scores
        st.session_state["tel_ml_labels"] = labels
        st.session_state["tel_ml_telemetry"] = telemetry
        st.session_state.pop("tel_ml_threshold", None)

    if "tel_ml_scores" not in st.session_state:
        st.info("Click **Run** to fit and score.")
        return
    _detector_results_section(
        "tel_ml",
        st.session_state["tel_ml_scores"],
        st.session_state["tel_ml_labels"],
        st.session_state["tel_ml_telemetry"],
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
    normal = telemetry[labels == 0]
    n_train = max(1, int(len(normal) * train_frac))
    train = normal[:n_train]

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

    normal = telemetry[labels == 0]
    n_train = max(1, len(normal) // 2)
    train = normal[:n_train]

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
    tabs = st.tabs(["Generator", "Statistical", "ML", "Deep", "Comparison"])
    with tabs[0]:
        tab_generator()
    with tabs[1]:
        tab_statistical()
    with tabs[2]:
        tab_ml()
    with tabs[3]:
        tab_deep()
    with tabs[4]:
        tab_comparison()


if __name__ == "__main__":
    main()
