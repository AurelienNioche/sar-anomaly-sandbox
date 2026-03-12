"""Shared disk I/O utilities for dashboard data loading and saving."""

import io
from datetime import datetime
from pathlib import Path

import torch


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
    tensors = tuple(
        torch.load(resolved / f, map_location="cpu", weights_only=True)
        for f in filenames
    )
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
