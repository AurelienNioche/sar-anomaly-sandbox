import io
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve

from src.data.generators import SpeckleSARGenerator
from src.data.generators.speckle import SpeckleSARGeneratorConfig
from src.models.baselines import RXDetector

st.set_page_config(page_title="SAR Anomaly Sandbox", layout="wide")

LABEL_NAMES = {0: "Normal", 1: "Anomaly"}
OUTCOME_COLORS = {"TP": "green", "TN": "blue", "FP": "orange", "FN": "red"}

GEN_DEFAULTS: dict = {
    "patch_size": 64,
    "n_looks": 4,
    "anomaly_ratio": 0.1,
    "anomaly_size": 3,
    "base_intensity": 1.0,
    "anomaly_intensity": 5.0,
    "seed": 42,
    "n_samples": 16,
}


def patch_to_display(patch: np.ndarray) -> np.ndarray:
    arr = np.squeeze(patch)
    if arr.ndim == 3:
        arr = arr[0]
    return np.log1p(np.clip(arr, 0, None))


def render_patch_grid(patches: torch.Tensor, labels: torch.Tensor, cols: int = 4) -> None:
    n = len(patches)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    for idx, ax in enumerate(axes.flat):
        if idx < n:
            arr = patch_to_display(patches[idx].numpy())
            ax.imshow(arr, cmap="gray")
            ax.set_title(LABEL_NAMES[int(labels[idx])])
        ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def outcome_label(true: int, pred: int) -> str:
    if true == 1 and pred == 1:
        return "TP"
    if true == 0 and pred == 0:
        return "TN"
    if true == 0 and pred == 1:
        return "FP"
    return "FN"


def render_patch_grid_with_outcomes(
    patches: torch.Tensor,
    true_labels: torch.Tensor,
    pred_labels: torch.Tensor,
    cols: int = 4,
    max_patches: int = 16,
) -> None:
    n = min(len(patches), max_patches)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    for idx, ax in enumerate(axes.flat):
        if idx < n:
            arr = patch_to_display(patches[idx].numpy())
            ax.imshow(arr, cmap="gray")
            oc = outcome_label(int(true_labels[idx]), int(pred_labels[idx]))
            ax.set_title(oc, color=OUTCOME_COLORS[oc], fontweight="bold")
        ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


DEFAULT_DATA_DIR = "data/synthetic"


def _find_latest_run(base: Path) -> Path | None:
    """Return the most recently modified subdirectory of *base* that contains
    both patches.pt and labels.pt, or None if none exists."""
    candidates = sorted(
        (d for d in base.iterdir() if d.is_dir()
         and (d / "patches.pt").exists()
         and (d / "labels.pt").exists()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_patches_labels(uploaded: list) -> tuple[torch.Tensor, torch.Tensor] | None:
    patches_file = None
    labels_file = None
    for f in uploaded:
        name = Path(f.name).name
        if name == "patches.pt":
            patches_file = f
        elif name == "labels.pt":
            labels_file = f
    if patches_file is None or labels_file is None:
        return None
    patches = torch.load(
        io.BytesIO(patches_file.getvalue()), map_location="cpu", weights_only=True
    )
    labels = torch.load(
        io.BytesIO(labels_file.getvalue()), map_location="cpu", weights_only=True
    )
    return patches, labels


def load_patches_labels_from_dir(
    dir_path: str,
) -> tuple[torch.Tensor, torch.Tensor, Path] | None:
    """Load patches and labels from *dir_path*.

    Timestamped run sub-folders take priority over flat files: if any valid
    sub-folder exists the most recently modified one is used, regardless of
    whether patches.pt / labels.pt also exist directly in dir_path.
    Returns (patches, labels, resolved_path) or None.
    """
    p = Path(dir_path)
    if not p.exists():
        return None
    latest = _find_latest_run(p)
    if latest is not None:
        resolved = latest
    elif (p / "patches.pt").exists() and (p / "labels.pt").exists():
        resolved = p
    else:
        return None
    patches = torch.load(resolved / "patches.pt", map_location="cpu", weights_only=True)
    labels = torch.load(resolved / "labels.pt", map_location="cpu", weights_only=True)
    return patches, labels, resolved


def data_source_widget(tab_key: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    dir_path = st.text_input(
        "Data directory",
        value=DEFAULT_DATA_DIR,
        key=f"{tab_key}_dir",
        help="Path to a folder containing patches.pt and labels.pt (or a parent "
             "with timestamped run sub-folders — the latest run is loaded automatically).",
    )
    result_tensors = None
    if dir_path:
        raw = load_patches_labels_from_dir(dir_path)
        if raw is not None:
            patches, labels, resolved = raw
            result_tensors = (patches, labels)
            resolved_str = str(resolved)
            if resolved_str != dir_path:
                st.success(f"Auto-selected latest run: `{resolved_str}`")
            else:
                st.success(f"Loaded from `{resolved_str}`")
        else:
            st.warning(f"`{dir_path}` not found or contains no valid run data.")

    uploaded = st.file_uploader(
        "Or drag-and-drop a folder to override",
        accept_multiple_files="directory",
        type=None,
        key=f"{tab_key}_upload",
    )
    if uploaded:
        result_tensors = load_patches_labels(uploaded)
        if result_tensors is None:
            st.error("Expected patches.pt and labels.pt in the selected folder.")
    return result_tensors


def save_run(
    patches: torch.Tensor,
    labels: torch.Tensor,
    base_dir: str = "data/synthetic",
) -> Path:
    """Save patches and labels to a timestamped sub-folder of *base_dir*."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = Path(base_dir) / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(patches, save_dir / "patches.pt")
    torch.save(labels, save_dir / "labels.pt")
    return save_dir


def _reset_generator_defaults() -> None:
    for key, val in GEN_DEFAULTS.items():
        st.session_state[key] = val


def tab_generator() -> None:
    st.header("Generator")
    st.markdown(
        "Experiment with synthetic SAR generator parameters and see the effect in real-time."
    )

    for key, val in GEN_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val

    col1, col2 = st.columns([1, 3])
    with col1:
        patch_size = st.slider("Patch size", 16, 128, 64, 16, key="patch_size")
        n_looks = st.slider("Number of looks", 1, 16, 4, key="n_looks")
        anomaly_ratio = st.slider("Anomaly ratio", 0.0, 0.5, 0.1, 0.05, key="anomaly_ratio")
        anomaly_size = st.slider("Anomaly size", 2, 8, 3, key="anomaly_size")
        base_intensity = st.slider(
            "Base intensity", 0.1, 2.0, 1.0, 0.1, key="base_intensity"
        )
        anomaly_intensity = st.slider(
            "Anomaly intensity", 1.0, 10.0, 5.0, 0.5, key="anomaly_intensity"
        )
        seed = st.number_input("Seed (optional)", min_value=0, value=42, step=1, key="seed")
        n_samples = st.slider("Samples to generate", 4, 32, 16, 4, key="n_samples")
        col_gen, col_reset = st.columns(2)
        with col_reset:
            st.button("Reset to defaults", on_click=_reset_generator_defaults)
        with col_gen:
            generate_clicked = st.button("Generate")
        if generate_clicked:
            config = SpeckleSARGeneratorConfig(
                patch_size=patch_size,
                n_looks=n_looks,
                anomaly_ratio=anomaly_ratio,
                anomaly_size=anomaly_size,
                base_intensity=base_intensity,
                anomaly_intensity=anomaly_intensity,
                seed=seed,
            )
            gen = SpeckleSARGenerator(config)
            patches, labels = gen.generate(n_samples)
            st.session_state["gen_patches"] = patches
            st.session_state["gen_labels"] = labels
            st.session_state["gen_saved_path"] = None

        if "gen_patches" in st.session_state:
            if st.button("Save to disk"):
                save_dir = save_run(
                    st.session_state["gen_patches"],
                    st.session_state["gen_labels"],
                )
                saved_path = str(save_dir)
                st.session_state["gen_saved_path"] = saved_path
                st.session_state["viz_dir"] = saved_path
                st.session_state["det_dir"] = saved_path

            if st.session_state.get("gen_saved_path"):
                st.success(
                    f"Saved to `{st.session_state['gen_saved_path']}` — "
                    "Visualize and Detector tabs updated automatically."
                )

    with col2:
        if "gen_patches" in st.session_state:
            render_patch_grid(
                st.session_state["gen_patches"],
                st.session_state["gen_labels"],
            )
        else:
            st.info("Click **Generate** to create samples with the current parameters.")


def tab_visualize() -> None:
    st.header("Visualize")
    st.markdown(
        "Type a directory path or drag-and-drop a folder containing "
        "`patches.pt` and `labels.pt` from `run_generate`."
    )

    result = data_source_widget("viz")
    if result is None:
        st.info("Select a data source above to visualize generated data.")
        return
    patches, labels = result

    n = len(patches)
    st.metric("Total patches", n)
    st.metric("Anomalies", int(labels.sum().item()))
    st.metric("Normal", n - int(labels.sum().item()))

    idx = st.slider("Patch index", 0, n - 1 if n > 0 else 0, 0)
    if n > 0:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        arr = patch_to_display(patches[idx].numpy())
        ax.imshow(arr, cmap="gray")
        ax.set_title(f"Patch {idx} — {LABEL_NAMES[int(labels[idx])]}")
        ax.axis("off")
        st.pyplot(fig)
        plt.close()

        st.subheader("Grid preview")
        preview_n = min(16, n)
        render_patch_grid(patches[:preview_n], labels[:preview_n], cols=4)


def _best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Return the threshold on the ROC curve that maximises F1."""
    _, _, thresholds = roc_curve(y_true, y_score)
    best_thresh = float(thresholds[0])
    best_f1 = 0.0
    for t in thresholds:
        preds = (y_score >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)
    return best_thresh


def tab_detector() -> None:
    st.header("Detector")
    st.markdown(
        "Type a directory path or drag-and-drop a folder, then fit the **RX detector** "
        "on normal patches and explore its performance."
    )

    result = data_source_widget("det")
    if result is None:
        st.info("Select a data source above to run the detector.")
        return
    patches, labels = result

    n_normal = int((labels == 0).sum().item())
    train_frac = st.slider(
        "Fraction of normal patches used for training",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
    )
    n_train = max(1, int(n_normal * train_frac))

    normal_patches = patches[labels == 0]
    train_patches = normal_patches[:n_train]

    st.caption(
        f"Training on {n_train} normal patches — testing on all {len(patches)} patches."
    )

    if st.button("Run RX Detector"):
        det = RXDetector().fit(train_patches)
        scores = det.score(patches)
        st.session_state["det_scores"] = scores
        st.session_state["det_labels"] = labels
        st.session_state["det_patches"] = patches
        st.session_state.pop("det_threshold", None)

    if "det_scores" not in st.session_state:
        st.info("Click **Run RX Detector** to fit and score.")
        return

    scores = st.session_state["det_scores"]
    labels = st.session_state["det_labels"]
    patches = st.session_state["det_patches"]
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
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Score Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(
            y_score[y_true == 0],
            bins=40,
            alpha=0.6,
            label="Normal",
            density=True,
        )
        ax.hist(
            y_score[y_true == 1],
            bins=40,
            alpha=0.6,
            label="Anomaly",
            density=True,
        )
        ax.set_xlabel("Anomaly score")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    default_threshold = st.session_state.get(
        "det_threshold", _best_f1_threshold(y_true, y_score)
    )
    st.subheader("Threshold")
    threshold = st.slider(
        "Decision threshold",
        min_value=float(y_score.min()),
        max_value=float(y_score.max()),
        value=float(np.clip(default_threshold, y_score.min(), y_score.max())),
        step=(float(y_score.max()) - float(y_score.min())) / 100,
        key="det_threshold",
    )
    preds = (y_score >= threshold).astype(int)
    col3, col4, col5 = st.columns(3)
    col3.metric("Precision", f"{precision_score(y_true, preds, zero_division=0):.3f}")
    col4.metric("Recall", f"{recall_score(y_true, preds, zero_division=0):.3f}")
    col5.metric("F1", f"{f1_score(y_true, preds, zero_division=0):.3f}")

    st.subheader("Patch grid — TP / TN / FP / FN")
    st.caption("Green=TP, Blue=TN, Orange=FP, Red=FN")
    pred_tensor = torch.tensor(preds)
    render_patch_grid_with_outcomes(patches, labels, pred_tensor, cols=4, max_patches=16)


def main() -> None:
    st.title("SAR Anomaly Sandbox")
    tab1, tab2, tab3 = st.tabs(["Generator", "Visualize", "Detector"])
    with tab1:
        tab_generator()
    with tab2:
        tab_visualize()
    with tab3:
        tab_detector()


if __name__ == "__main__":
    main()
