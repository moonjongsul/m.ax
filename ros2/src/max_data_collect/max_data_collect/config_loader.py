"""Read the flat ROS parameter tree back into a nested dict.

rclpy flattens the YAML into dotted names ('storage.video_codec'); this
rebuilds the nesting, applies defaults, and fails loudly on anything the
recorder cannot run without. Validating here means a typo in the YAML is
a startup error rather than a KeyError twenty minutes into a session.
"""

REQUIRED_SECTIONS = (
    "recording", "storage", "task", "qos", "record_interface",
    "action", "observation_cameras", "observation_manipulator",
    "observation_gripper",
)

DEFAULTS = {
    "source": "demo",
    "recording": {
        "collect_hz": 30.0,
        "warmup_sec": 1.0,
        "staleness_timeout": 0.5,
        "require_all_sources": True,
        "queue_size": 900,
        "on_overflow": "abort",
    },
    "storage": {
        "data_format": "hdf5",
        "video_codec": "hevc_nvenc",
        "video_cq": 28,
        "video_preset": "p4",
        "async_save": True,
        "undo_depth": 1,
        "resume_existing": True,
        "keyframes": {
            "enabled": True,
            "jpeg_quality": 95,
            "composite_order": [],
        },
    },
    "task": {
        "main_prompt": "",
        "goal_image": "",
        "total_score": 1.0,
        "segment_sec": 1.0,
    },
    "qos": {"reliability": "best_effort", "depth": 10},
    "record_interface": {"status_period": 1.0},
}


def _unflatten(flat: dict) -> dict:
    out: dict = {}
    for name, value in flat.items():
        node = out
        parts = name.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def _merge_defaults(cfg: dict, defaults: dict) -> None:
    for key, value in defaults.items():
        if isinstance(value, dict):
            _merge_defaults(cfg.setdefault(key, {}), value)
        else:
            cfg.setdefault(key, value)


def load(node) -> dict:
    """Build the validated config dict from a node's parameters."""
    flat = {k: p.value for k, p in node.get_parameters_by_prefix("").items()}
    # rclpy injects its own parameters into the same namespace.
    for reserved in ("use_sim_time", "start_type_description_service"):
        flat.pop(reserved, None)

    cfg = _unflatten(flat)
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise RuntimeError(
            f"config is missing section(s) {missing}; pass the YAML with "
            "--params-file or use the launch file"
        )
    _merge_defaults(cfg, DEFAULTS)

    for key in ("output_dir", "dataset_name"):
        if not cfg.get(key):
            raise RuntimeError(f"config is missing '{key}'")

    hz = float(cfg["recording"]["collect_hz"])
    if hz <= 0.0:
        raise RuntimeError(f"recording.collect_hz must be > 0, got {hz}")
    cfg["recording"]["collect_hz"] = hz

    if cfg["storage"]["data_format"] != "hdf5":
        raise RuntimeError(
            f"storage.data_format '{cfg['storage']['data_format']}' is not "
            "supported; only 'hdf5' is implemented"
        )
    if cfg["recording"]["on_overflow"] not in ("abort", "drop"):
        raise RuntimeError(
            "recording.on_overflow must be 'abort' or 'drop', got "
            f"'{cfg['recording']['on_overflow']}'"
        )

    if not cfg["observation_cameras"]:
        raise RuntimeError("config lists no cameras under observation_cameras")
    for name, cam in cfg["observation_cameras"].items():
        if not cam.get("topic"):
            raise RuntimeError(f"camera '{name}' has no topic")
        res = list(cam.get("resolution") or [0, 0])
        if len(res) != 2:
            raise RuntimeError(
                f"camera '{name}': resolution must be [width, height], got {res}"
            )
        cam["resolution"] = [int(res[0]), int(res[1])]
        cam["rotate"] = int(cam.get("rotate", 0)) % 360
        if cam["rotate"] not in (0, 90, 180, 270):
            raise RuntimeError(
                f"camera '{name}': rotate must be 0/90/180/270, got {cam['rotate']}"
            )

    keyframes = cfg["storage"]["keyframes"]
    keyframes["enabled"] = bool(keyframes["enabled"])
    quality = int(keyframes["jpeg_quality"])
    if not 1 <= quality <= 100:
        raise RuntimeError(
            f"storage.keyframes.jpeg_quality must be 1..100, got {quality}"
        )
    keyframes["jpeg_quality"] = quality
    # Empty means "every camera, in observation_cameras order"; naming a
    # camera that does not exist is a typo worth failing at startup for.
    order = list(keyframes["composite_order"] or cfg["observation_cameras"])
    unknown = [n for n in order if n not in cfg["observation_cameras"]]
    if unknown:
        raise RuntimeError(
            f"storage.keyframes.composite_order names unconfigured camera(s) "
            f"{unknown}; known cameras are {list(cfg['observation_cameras'])}"
        )
    keyframes["composite_order"] = order

    manip = cfg["observation_manipulator"]
    for block in ("joint", "eef_position", "eef_velocity"):
        if not manip.get(block, {}).get("topic"):
            raise RuntimeError(f"observation_manipulator.{block} has no topic")
    if not manip["joint"].get("joint_names"):
        raise RuntimeError("observation_manipulator.joint.joint_names is empty")
    manip["joint"]["joint_names"] = list(manip["joint"]["joint_names"])

    grip = cfg["observation_gripper"]
    if not grip.get("topic"):
        raise RuntimeError("observation_gripper has no topic")
    if not grip.get("joint_name"):
        raise RuntimeError("observation_gripper.joint_name is empty")
    effort = grip.setdefault("effort", {})
    lo = float(effort.get("effort_min", 0.0))
    hi = float(effort.get("effort_max", 600.0))
    if hi <= lo:
        raise RuntimeError(
            f"observation_gripper.effort: effort_max ({hi}) must exceed "
            f"effort_min ({lo})"
        )
    effort["effort_min"], effort["effort_max"] = lo, hi

    if not cfg["action"].get("topic"):
        raise RuntimeError("action has no topic")

    # Windows are stride-1, so segment index == frame index and nothing
    # segment-shaped is stored; this is only a hint for the editing GUI.
    seg_sec = float(cfg["task"]["segment_sec"])
    cfg["task"]["segment_sec"] = seg_sec
    cfg["task"]["segment_frames"] = max(1, round(seg_sec * hz))

    return cfg
