"""Load and validate editor_config.yaml.

Plain YAML rather than a ROS parameter file: the editor is a standalone
app, so `max_data_collect.config_loader` (which reads rclpy parameters)
does not apply. The intent is the same though -- validate at startup so a
typo in the YAML is a startup error, not a surprise mid-session.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml

# <repo>/ros2/src/max_data_collect/max_data_collect/editor/config.py -> <repo>
REPO_ROOT = Path(__file__).resolve().parents[5]
PKG_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PKG_ROOT / "config" / "editor_config.yaml"

PROXY_MODES = ("auto", "never", "always")
REDUCE_MODES = ("components", "scalar", "norm3", "norm")

DEFAULTS: dict[str, Any] = {
    "dataset": {"root": "datasets/xarm7", "default_dataset": ""},
    "server": {"host": "0.0.0.0", "port": 8020},
    "video": {
        "proxy": "auto",
        "playable_codecs": ["h264", "avc1", "vp8", "vp9", "av1"],
        "cache_dirname": ".proxy_cache",
        "encoder": "h264_nvenc",
        "nvenc_preset": "p4",
        "nvenc_cq": 26,
        "x264_preset": "veryfast",
        "x264_crf": 23,
    },
    "timeline": {
        "max_points": 1500,
        "channels": [
            {"key": "action.gripper", "label": "gripper cmd", "reduce": "components"},
            {"key": "observation.gripper_position", "label": "gripper pos", "reduce": "components"},
            {"key": "observation.gripper_effort", "label": "gripper effort", "reduce": "components"},
            {"key": "observation.joint_position", "label": "joint pos", "reduce": "components"},
            {"key": "observation.joint_velocity", "label": "joint vel", "reduce": "components"},
            {"key": "observation.eef_position", "label": "eef pos", "reduce": "components"},
            {"key": "observation.eef_orientation", "label": "eef quat", "reduce": "components"},
            {"key": "observation.eef_velocity", "label": "eef vel", "reduce": "components"},
            {"key": "action.cartesian_velocity", "label": "cmd vel", "reduce": "components"},
        ],
    },
    "autosegment": {
        "gripper_key": "action.gripper",
        "gripper_min_span": 0.2,
        "idle_speed_ratio": 0.05,
        "trim_margin_frames": 5,
        "min_segment_frames": 5,
        "close_lead_sec": 0.5,
        "open_lag_sec": 1.0,
    },
    "labels": {"default_vocabulary": []},
    "subtasks": {},
    "objects": [],
    "targets": [],
    "edits": {"history_depth": 5},
}


def _merge_defaults(cfg: dict, defaults: dict) -> None:
    for key, value in defaults.items():
        if isinstance(value, dict):
            _merge_defaults(cfg.setdefault(key, {}), value)
        else:
            cfg.setdefault(key, value)


def load(path: str | Path | None = None) -> dict[str, Any]:
    """Return the validated config. A missing file yields the defaults."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if path and not cfg_path.is_file():
        raise RuntimeError(f"config file not found: {cfg_path}")

    raw: dict[str, Any] = {}
    if cfg_path.is_file():
        loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if loaded is not None and not isinstance(loaded, dict):
            raise RuntimeError(f"{cfg_path}: top level must be a mapping")
        raw = loaded or {}
        if "ros__parameters" in raw or any(
            isinstance(v, dict) and "ros__parameters" in v for v in raw.values()
        ):
            raise RuntimeError(
                f"{cfg_path} looks like a ROS parameter file; editor_config.yaml "
                "is plain YAML with no 'ros__parameters:' root"
            )

    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise RuntimeError(
            f"{cfg_path}: unknown section(s) {sorted(unknown)}; "
            f"known sections are {sorted(DEFAULTS)}"
        )

    cfg = copy.deepcopy(raw)
    _merge_defaults(cfg, DEFAULTS)
    cfg["_path"] = str(cfg_path) if cfg_path.is_file() else None
    _validate(cfg, cfg_path)
    return cfg


def resolve_root(cfg: dict[str, Any]) -> Path:
    """Dataset root as an absolute path (relative values hang off the repo root)."""
    root = Path(str(cfg["dataset"]["root"])).expanduser()
    return root if root.is_absolute() else (REPO_ROOT / root).resolve()


def _validate(cfg: dict[str, Any], where: Path) -> None:
    video = cfg["video"]
    if video["proxy"] not in PROXY_MODES:
        raise RuntimeError(
            f"{where}: video.proxy must be one of {PROXY_MODES}, got '{video['proxy']}'"
        )
    video["playable_codecs"] = [str(c).lower() for c in video["playable_codecs"] or []]
    if not str(video["cache_dirname"]).strip():
        raise RuntimeError(f"{where}: video.cache_dirname must not be empty")
    if "/" in str(video["cache_dirname"]):
        raise RuntimeError(f"{where}: video.cache_dirname must be a single directory name")

    port = int(cfg["server"]["port"])
    if not 1 <= port <= 65535:
        raise RuntimeError(f"{where}: server.port out of range: {port}")
    cfg["server"]["port"] = port

    points = int(cfg["timeline"]["max_points"])
    if not 100 <= points <= 20000:
        raise RuntimeError(f"{where}: timeline.max_points must be 100..20000, got {points}")
    cfg["timeline"]["max_points"] = points

    channels = cfg["timeline"]["channels"]
    if not isinstance(channels, list) or not channels:
        raise RuntimeError(f"{where}: timeline.channels must be a non-empty list")
    for ch in channels:
        if not isinstance(ch, dict) or not ch.get("key"):
            raise RuntimeError(f"{where}: every timeline.channels entry needs a 'key'")
        ch.setdefault("label", ch["key"])
        ch["reduce"] = ch.get("reduce", "components")
        if ch["reduce"] not in REDUCE_MODES:
            raise RuntimeError(
                f"{where}: channel '{ch['key']}' has reduce='{ch['reduce']}'; "
                f"must be one of {REDUCE_MODES}"
            )

    seg = cfg["autosegment"]
    for key in ("gripper_min_span", "idle_speed_ratio"):
        value = float(seg[key])
        if not 0.0 <= value <= 1.0:
            raise RuntimeError(f"{where}: autosegment.{key} must be 0..1, got {value}")
        seg[key] = value
    for key in ("trim_margin_frames", "min_segment_frames"):
        value = int(seg[key])
        if value < 0:
            raise RuntimeError(f"{where}: autosegment.{key} must be >= 0, got {value}")
        seg[key] = value

    # Offsets are in seconds and converted with the dataset's collect_hz.
    for key in ("close_lead_sec", "open_lag_sec"):
        value = float(seg[key])
        if not 0.0 <= value <= 5.0:
            raise RuntimeError(f"{where}: autosegment.{key} must be 0..5 seconds, got {value}")
        seg[key] = value

    depth = int(cfg["edits"]["history_depth"])
    if depth < 0:
        raise RuntimeError(f"{where}: edits.history_depth must be >= 0, got {depth}")
    cfg["edits"]["history_depth"] = depth

    vocab = cfg["labels"]["default_vocabulary"] or []
    if not isinstance(vocab, list):
        raise RuntimeError(f"{where}: labels.default_vocabulary must be a list")
    cfg["labels"]["default_vocabulary"] = [str(v).strip() for v in vocab if str(v).strip()]

    _validate_subtasks(cfg, where)


def _validate_subtasks(cfg: dict[str, Any], where: Path) -> None:
    """subtasks/objects/targets: prompt templates over a small vocabulary.

    A template is a label key mapped to a sentence that may contain the
    placeholders {object} and {target}. The UI expands it into the segment
    `prompt` while the key itself becomes the segment `label`, so the short
    tag stays stable for stats while the sentence is what a VLA model reads.
    """
    subtasks = cfg["subtasks"] or {}
    if not isinstance(subtasks, dict):
        raise RuntimeError(f"{where}: subtasks must be a mapping of label -> template")

    for key in ("objects", "targets"):
        items = cfg[key] or []
        if not isinstance(items, list):
            raise RuntimeError(f"{where}: {key} must be a list")
        cfg[key] = [str(v).strip() for v in items if str(v).strip()]

    clean: dict[str, str] = {}
    for label, template in subtasks.items():
        label = str(label).strip()
        template = str(template or "").strip()
        if not label or not template:
            raise RuntimeError(f"{where}: subtasks entry '{label}' has an empty label or template")
        fields = set(re.findall(r"{(\w+)}", template))
        unknown = fields - {"object", "target"}
        if unknown:
            raise RuntimeError(
                f"{where}: subtasks['{label}'] uses unknown placeholder(s) "
                f"{sorted(unknown)}; only {{object}} and {{target}} are supported"
            )
        # A template that asks for a slot with nothing to fill it can never be
        # expanded, so it is a config error rather than a silent empty dropdown.
        for field, pool in (("object", "objects"), ("target", "targets")):
            if field in fields and not cfg[pool]:
                raise RuntimeError(
                    f"{where}: subtasks['{label}'] uses {{{field}}} but '{pool}' is empty"
                )
        clean[label] = template
    cfg["subtasks"] = clean
