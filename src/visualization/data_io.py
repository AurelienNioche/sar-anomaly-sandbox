"""Shared disk I/O utilities for dashboard data loading and saving."""

import io
import json
from datetime import datetime
from pathlib import Path

import torch


def load_tensor(path: Path) -> torch.Tensor:
    """Load a single tensor from *path* with safe defaults."""
    return torch.load(path, map_location="cpu", weights_only=True)


def list_runs(base_dir: str, filenames: tuple[str, ...]) -> list[Path]:
    """Return all valid run directories inside *base_dir*, newest first.

    A directory is valid if it contains every file in *filenames*.
    """
    base = Path(base_dir)
    if not base.exists():
        return []
    return sorted(
        [
            d for d in base.iterdir()
            if d.is_dir() and all((d / f).exists() for f in filenames)
        ],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )


def _find_latest_run(base: Path) -> Path | None:
    """Return the most recently modified sub-directory of *base* that contains
    both patches.pt (or telemetry.pt) and labels.pt, or None if none exists."""
    candidates = sorted(
        (
            d for d in base.iterdir()
            if d.is_dir()
            and (d / "labels.pt").exists()
            and ((d / "patches.pt").exists() or (d / "telemetry.pt").exists())
        ),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_tensors_from_dir(
    dir_path: str,
    filenames: tuple[str, ...] = ("patches.pt", "labels.pt"),
) -> tuple[tuple[torch.Tensor, ...], Path] | None:
    """Load *filenames* from *dir_path*, preferring the latest timestamped run.

    Returns (tuple_of_tensors, resolved_path) or None.
    """
    p = Path(dir_path)
    if not p.exists():
        return None
    latest = _find_latest_run(p)
    if latest is not None:
        resolved = latest
    elif all((p / f).exists() for f in filenames):
        resolved = p
    else:
        return None
    tensors = tuple(load_tensor(resolved / f) for f in filenames)
    return tensors, resolved


def load_tensors_from_upload(
    uploaded: list,
    filenames: tuple[str, ...] = ("patches.pt", "labels.pt"),
) -> tuple[torch.Tensor, ...] | None:
    """Load *filenames* from a Streamlit file-uploader result."""
    mapping: dict[str, object] = {}
    for f in uploaded:
        name = Path(f.name).name
        if name in filenames:
            mapping[name] = f
    if any(n not in mapping for n in filenames):
        return None
    return tuple(
        torch.load(io.BytesIO(mapping[n].getvalue()), map_location="cpu", weights_only=True)
        for n in filenames
    )


def save_run(
    tensors: dict[str, torch.Tensor],
    base_dir: str,
) -> Path:
    """Save *tensors* (filename -> tensor) to a timestamped sub-folder of *base_dir*."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = Path(base_dir) / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    for filename, tensor in tensors.items():
        torch.save(tensor, save_dir / filename)
    return save_dir


# ---------------------------------------------------------------------------
# Detector-run persistence
# ---------------------------------------------------------------------------

def save_detector_run(
    dataset_dir: Path,
    tab_key: str,
    det_name: str,
    scores_test: torch.Tensor,
    labels_mc_test: torch.Tensor,
    scores_train: torch.Tensor,
    labels_mc_train: torch.Tensor,
    split_info: dict,
    test_tel: torch.Tensor | None = None,
    test_labels_mc: torch.Tensor | None = None,
    hyperparams: dict | None = None,
) -> Path:
    """Persist detector scores and metadata under ``<dataset_dir>/detector_runs/``.

    The ``metadata.json`` embeds ``dataset_run`` (the dataset directory name)
    so that a run can always be matched back to the exact dataset it came from.

    Returns the path of the newly created run directory.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    run_dir = dataset_dir / "detector_runs" / f"{timestamp}_{tab_key}"
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(scores_test,    run_dir / "scores_test.pt")
    torch.save(labels_mc_test, run_dir / "labels_mc_test.pt")
    torch.save(scores_train,   run_dir / "scores_train.pt")
    torch.save(labels_mc_train, run_dir / "labels_mc_train.pt")
    if test_tel is not None:
        torch.save(test_tel,       run_dir / "test_tel.pt")
    if test_labels_mc is not None:
        torch.save(test_labels_mc, run_dir / "test_labels_mc.pt")

    metadata: dict = {
        "detector_name": det_name,
        "tab_key":       tab_key,
        "dataset_run":   dataset_dir.name,
        "split_info":    split_info,
        "timestamp":     datetime.now().isoformat(),
        "hyperparams":   hyperparams or {},
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return run_dir


def list_detector_runs(dataset_dir: Path) -> list[dict]:
    """Return saved detector runs for *dataset_dir*, newest first.

    Each entry is a dict with:
      ``metadata``         – parsed metadata.json
      ``scores_test``      – (M*T,) tensor
      ``labels_mc_test``   – (M*T,) tensor
      ``scores_train``     – (K*T,) tensor
      ``labels_mc_train``  – (K*T,) tensor
      ``test_tel``         – (M, T, C) tensor  (optional)
      ``test_labels_mc``   – (M, T) tensor     (optional)

    Entries whose metadata ``dataset_run`` does not match
    ``dataset_dir.name`` are silently skipped (stale / misplaced files).
    """
    runs_dir = dataset_dir / "detector_runs"
    if not runs_dir.exists():
        return []

    result: list[dict] = []
    for d in sorted(runs_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            metadata = json.loads(meta_path.read_text())
        except Exception:
            continue
        if metadata.get("dataset_run") != dataset_dir.name:
            continue

        entry: dict = {"metadata": metadata}
        for fname, key in [
            ("scores_test.pt",    "scores_test"),
            ("labels_mc_test.pt", "labels_mc_test"),
            ("scores_train.pt",   "scores_train"),
            ("labels_mc_train.pt","labels_mc_train"),
            ("test_tel.pt",       "test_tel"),
            ("test_labels_mc.pt", "test_labels_mc"),
        ]:
            path = d / fname
            if path.exists():
                try:
                    entry[key] = torch.load(path, map_location="cpu", weights_only=True)
                except Exception:
                    pass
        result.append(entry)
    return result
