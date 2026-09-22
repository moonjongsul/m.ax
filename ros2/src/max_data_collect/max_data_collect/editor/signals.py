"""HDF5 timeseries extraction and heuristics that seed segment boundaries."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .store import DEFAULT_SEGMENT_SCORE

# Fallback when no channel list is configured: every vector expanded into
# its individual components, so nothing in the HDF5 is silently hidden.
_PLOT_SPEC = [
    {"key": "action.gripper", "label": "gripper cmd", "reduce": "components"},
    {"key": "observation.gripper_position", "label": "gripper pos", "reduce": "components"},
    {"key": "observation.gripper_effort", "label": "gripper effort", "reduce": "components"},
    {"key": "observation.joint_position", "label": "joint pos", "reduce": "components"},
    {"key": "observation.joint_velocity", "label": "joint vel", "reduce": "components"},
    {"key": "observation.eef_position", "label": "eef pos", "reduce": "components"},
    {"key": "observation.eef_orientation", "label": "eef quat", "reduce": "components"},
    {"key": "observation.eef_velocity", "label": "eef vel", "reduce": "components"},
    {"key": "action.cartesian_velocity", "label": "cmd vel", "reduce": "components"},
]

# Per-component suffixes, chosen by (key, width) so a 6-vector of twist
# reads "vx vy vz wx wy wz" rather than "0 1 2 3 4 5".
_AXIS_NAMES = {
    3: ["x", "y", "z"],
    4: ["qx", "qy", "qz", "qw"],
    6: ["vx", "vy", "vz", "wx", "wy", "wz"],
}
_ROT6D_NAMES = ["r11", "r21", "r31", "r12", "r22", "r32"]


def _component_names(key: str, width: int, joint_names: list[str] | None) -> list[str]:
    if width == 1:
        return [""]
    if "joint" in key:
        # Real joint names from metadata when they line up, else joint1..N.
        if joint_names and len(joint_names) == width:
            return list(joint_names)
        return [f"joint{i + 1}" for i in range(width)]
    if "rot6d" in key:
        return _ROT6D_NAMES[:width] if width <= len(_ROT6D_NAMES) else [str(i) for i in range(width)]
    if "position" in key and width == 3:
        return _AXIS_NAMES[3]
    return _AXIS_NAMES.get(width) or [str(i) for i in range(width)]


def _open(path: Path):
    import h5py  # imported lazily so the app still starts without h5py

    return h5py.File(path, "r")


def _reduce(arr: np.ndarray, mode: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    if mode == "norm3" and arr.shape[1] >= 3:
        return np.linalg.norm(arr[:, :3], axis=1)
    if mode == "norm":
        return np.linalg.norm(arr, axis=1)
    return arr[:, 0]


def _series_for(key: str, raw: np.ndarray, mode: str,
                joint_names: list[str] | None) -> list[tuple[str, np.ndarray]]:
    """Expand one HDF5 dataset into the (suffix, values) traces it contributes."""
    arr = np.asarray(raw, dtype=np.float64)
    if mode != "components" or arr.ndim == 1:
        return [("", _reduce(arr, mode))]
    names = _component_names(key, arr.shape[1], joint_names)
    return [(names[i], arr[:, i]) for i in range(arr.shape[1])]


def _downsample(values: np.ndarray, max_points: int) -> list[float | None]:
    """Min/max-preserving decimation so spikes survive the trip to the browser."""
    n = values.size
    if n <= max_points:
        out = values
    else:
        step = int(np.ceil(n / max_points))
        trimmed = values[: (n // step) * step].reshape(-1, step)
        out = trimmed.mean(axis=1)
    return [None if not np.isfinite(v) else round(float(v), 5) for v in out]


def read_series(h5_path: Path, max_points: int = 1500,
                spec: list[dict[str, Any]] | None = None,
                joint_names: list[str] | None = None) -> dict[str, Any]:
    """Return plot-ready channels plus the frame stride used for decimation.

    A `components` channel yields one trace per vector element, each tagged
    with `group` (the source dataset) so the UI can stack them together.
    """
    spec = spec or _PLOT_SPEC
    with _open(h5_path) as f:
        available = set(f.keys())
        channels = []
        num_frames = 0
        for entry in spec:
            key = entry["key"]
            label = entry.get("label", key)
            if key not in available:
                continue
            mode = entry.get("reduce", "scalar")
            for suffix, values in _series_for(key, f[key][:], mode, joint_names):
                num_frames = max(num_frames, values.size)
                channels.append({
                    "key": key if not suffix else f"{key}[{suffix}]",
                    "label": f"{label} {suffix}".strip(),
                    "group": label,
                    "component": suffix,
                    "values": _downsample(values, max_points),
                    "min": round(float(np.nanmin(values)), 5) if values.size else 0.0,
                    "max": round(float(np.nanmax(values)), 5) if values.size else 0.0,
                })
        stale = f["stale"][:].astype(np.int64) if "stale" in available else np.zeros(0, dtype=np.int64)
        timestamps = f["timestamp"][:] if "timestamp" in available else np.zeros(0)

    stride = max(1, int(np.ceil(num_frames / max_points))) if num_frames else 1
    duration = float(timestamps[-1] - timestamps[0]) if timestamps.size > 1 else 0.0
    return {
        "num_frames": int(num_frames),
        "stride": stride,
        "duration": round(duration, 4),
        "channels": channels,
        "stale_frames": [int(i) for i in np.flatnonzero(stale)][:2000],
    }


def suggest_segments(h5_path: Path, cfg: dict[str, Any] | None = None,
                     fps: float = 30.0) -> dict[str, Any]:
    """Propose segment cuts and trim bounds from gripper transitions and idle spans.

    For pick-and-place, gripper open/close edges line up almost exactly with
    subtask boundaries, so they make a good first pass for a human to refine.
    Thresholds come from the `autosegment` config section.

    The raw gripper edge is not quite the right cut, though: the approach that
    precedes a grasp and the retreat that follows a release belong to the
    neighbouring subtask. So a closing edge (open->closed) is pulled EARLIER by
    `close_lead_sec` and an opening edge (closed->open) is pushed LATER by
    `open_lag_sec`, both converted to frames via `fps`.
    """
    cfg = cfg or {}
    min_frames = int(cfg.get("min_segment_frames", 5))
    min_span = float(cfg.get("gripper_min_span", 0.2))
    idle_ratio = float(cfg.get("idle_speed_ratio", 0.05))
    margin = int(cfg.get("trim_margin_frames", 5))
    preferred = str(cfg.get("gripper_key", "action.gripper"))
    fps = float(fps) if fps and fps > 0 else 30.0
    close_lead = int(round(float(cfg.get("close_lead_sec", 0.5)) * fps))
    open_lag = int(round(float(cfg.get("open_lag_sec", 0.5)) * fps))

    with _open(h5_path) as f:
        keys = set(f.keys())
        grip_key = preferred if preferred in keys else "observation.gripper_position"
        grip = np.asarray(f[grip_key][:], dtype=np.float64).ravel() if grip_key in keys else np.zeros(0)
        vel = (
            _reduce(f["observation.eef_velocity"][:], "norm3")
            if "observation.eef_velocity" in keys
            else np.zeros(grip.size)
        )

    n = int(grip.size or vel.size)
    if n == 0:
        return {"boundaries": [], "segments": [], "suggested_trim": {"start": 0, "end": 0}, "transitions": []}

    # Gripper edges: binarize around the mid-point of its observed range.
    transitions: list[int] = []
    raw_transitions: list[int] = []
    if grip.size:
        lo, hi = float(np.nanmin(grip)), float(np.nanmax(grip))
        if hi - lo > min_span:
            closed = grip < (lo + hi) / 2.0
            step = np.diff(closed.astype(np.int8))
            edges = np.flatnonzero(step != 0) + 1
            raw_transitions = [int(i) for i in edges]
            # step > 0 means the gripper just closed -> cut before the grasp;
            # step < 0 means it just opened -> cut after the release.
            for i in edges:
                shift = -close_lead if step[i - 1] > 0 else open_lag
                transitions.append(int(np.clip(i + shift, 0, n - 1)))
            # Shifting can reorder or collide two nearby edges.
            transitions = sorted(set(transitions))

    # Idle head/tail: leading and trailing frames where the arm barely moves.
    start, end = 0, n
    if vel.size:
        moving_thresh = max(1e-3, idle_ratio * float(np.nanmax(vel)))
        moving = np.flatnonzero(vel > moving_thresh)
        if moving.size:
            start = int(max(0, moving[0] - margin))
            end = int(min(n, moving[-1] + margin + 1))

    bounds = sorted({start, end} | {t for t in transitions if start < t < end})
    segments = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a >= min_frames:
            segments.append({"start": int(a), "end": int(b), "label": "", "prompt": "",
                             "score": DEFAULT_SEGMENT_SCORE})

    return {
        "boundaries": bounds,
        "segments": segments,
        "suggested_trim": {"start": start, "end": end},
        "transitions": transitions,
        "raw_transitions": raw_transitions,
    }
