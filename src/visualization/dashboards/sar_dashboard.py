import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from src.data.generators import SpeckleSARGenerator
from src.data.generators.speckle import SpeckleSARGeneratorConfig

st.set_page_config(page_title="SAR Anomaly Sandbox", layout="wide")

LABEL_NAMES = {0: "Normal", 1: "Anomaly"}


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


def tab_generator() -> None:
    st.header("Generator")
    st.markdown(
        "Experiment with synthetic SAR generator parameters and see the effect in real-time."
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        patch_size = st.slider("Patch size", 16, 128, 64, 16)
        n_looks = st.slider("Number of looks", 1, 16, 4)
        anomaly_ratio = st.slider("Anomaly ratio", 0.0, 0.5, 0.1, 0.05)
        anomaly_size = st.slider("Anomaly size", 2, 8, 3)
        base_intensity = st.slider("Base intensity", 0.1, 2.0, 1.0, 0.1)
        anomaly_intensity = st.slider("Anomaly intensity", 1.0, 5.0, 3.0, 0.5)
        seed = st.number_input("Seed (optional)", min_value=0, value=42, step=1)
        n_samples = st.slider("Samples to generate", 4, 32, 16, 4)
        if st.button("Generate"):
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
        "Drag and drop a folder (or browse) containing `patches.pt` and `labels.pt` "
        "from `run_generate`."
    )

    uploaded = st.file_uploader(
        "Drop folder or browse",
        accept_multiple_files="directory",
        type=None,
        help="Select a folder with patches.pt and labels.pt",
    )

    if not uploaded:
        st.info("Select a folder to visualize generated data.")
        return

    patches_file = None
    labels_file = None
    for f in uploaded:
        name = Path(f.name).name
        if name == "patches.pt":
            patches_file = f
        elif name == "labels.pt":
            labels_file = f

    if patches_file is None or labels_file is None:
        st.error("Expected patches.pt and labels.pt in the selected folder.")
        return

    patches = torch.load(
        io.BytesIO(patches_file.getvalue()),
        map_location="cpu",
        weights_only=True,
    )
    labels = torch.load(
        io.BytesIO(labels_file.getvalue()),
        map_location="cpu",
        weights_only=True,
    )

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


def main() -> None:
    st.title("SAR Anomaly Sandbox")
    tab1, tab2 = st.tabs(["Generator", "Visualize"])
    with tab1:
        tab_generator()
    with tab2:
        tab_visualize()


if __name__ == "__main__":
    main()
