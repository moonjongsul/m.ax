"""Dataset directory layout, episode numbering, and metadata.json.

    <output_dir>/<dataset_name>/
    ├─ metadata.json                 dataset-level shell
    └─ episodes/episode_000001/
       ├─ data.hdf5
       ├─ tasks.json                 per-episode prompts / scores
       └─ <camera>.mp4

Appending to an existing dataset resumes from the highest existing
episode number + 1 -- not count + 1, so a gap left by a deleted episode
can never cause an overwrite. A mismatch against the existing
metadata.json is warned about, not refused.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

EPISODE_RE = re.compile(r"^episode_(\d{6})$")

# Mismatches worth telling the operator about when appending. Everything
# else is per-episode and may legitimately differ.
COMPARED_FIELDS = ("collect_hz", "robot", "gripper", "cameras")


class Dataset:
    """Owns the dataset directory and hands out episode numbers."""

    def __init__(self, logger, cfg: dict):
        self._log = logger
        self._cfg = cfg
        self.root = Path(cfg["output_dir"]) / cfg["dataset_name"]
        self.episodes_dir = self.root / "episodes"
        self.metadata_path = self.root / "metadata.json"
        self._next_index = 0

    # ── setup ──────────────────────────────────────────────────────────

    def open(self) -> None:
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        existing = self._scan()
        self._next_index = (max(existing) + 1) if existing else 0

        if self.metadata_path.exists():
            self._warn_on_mismatch()
            self._log.info(
                f"[collect] appending to {self.root} "
                f"({len(existing)} episode(s) present, next is "
                f"episode_{self._next_index:06d})"
            )
        else:
            self._write_metadata({"episodes": []})
            self._log.info(f"[collect] created dataset {self.root}")

    def _scan(self) -> list[int]:
        """Episode numbers already on disk, however incomplete."""
        found = []
        for path in self.episodes_dir.iterdir():
            if not path.is_dir():
                continue
            m = EPISODE_RE.match(path.name)
            if not m:
                continue
            found.append(int(m.group(1)))
            if not (path / "data.hdf5").exists():
                self._log.warning(
                    f"[collect] {path.name} has no data.hdf5 -- left over from "
                    "an interrupted run; its number will not be reused"
                )
        return found

    def _warn_on_mismatch(self) -> None:
        try:
            existing = json.loads(self.metadata_path.read_text())
        except (OSError, ValueError) as exc:
            self._log.warning(f"[collect] cannot read metadata.json: {exc}")
            return
        current = self._describe()
        for field in COMPARED_FIELDS:
            if field in existing and existing[field] != current[field]:
                self._log.warning(
                    f"[collect] metadata.json '{field}' differs from the "
                    f"current config: existing={existing[field]!r} "
                    f"current={current[field]!r} -- collecting anyway, but the "
                    "episodes in this dataset will not be uniform"
                )

    # ── episodes ───────────────────────────────────────────────────────

    def allocate(self) -> tuple[int, str, Path]:
        """Reserve the next episode number and return its directory."""
        index = self._next_index
        episode_id = f"episode_{index:06d}"
        path = self.episodes_dir / episode_id
        while path.exists():
            index += 1
            episode_id = f"episode_{index:06d}"
            path = self.episodes_dir / episode_id
        self._next_index = index + 1
        return index, episode_id, path

    def register(self, entry: dict) -> None:
        """Append a finished episode to metadata.json."""
        meta = self._describe()
        meta["episodes"] = []
        if self.metadata_path.exists():
            try:
                meta["episodes"] = json.loads(
                    self.metadata_path.read_text()
                ).get("episodes", [])
            except (OSError, ValueError) as exc:
                self._log.warning(f"[collect] metadata.json unreadable: {exc}")
        meta["episodes"] = [
            e for e in meta["episodes"] if e.get("id") != entry["id"]
        ]
        meta["episodes"].append(entry)
        self._write_metadata(meta)

    def unregister(self, episode_id: str) -> None:
        if not self.metadata_path.exists():
            return
        try:
            meta = json.loads(self.metadata_path.read_text())
        except (OSError, ValueError):
            return
        meta["episodes"] = [
            e for e in meta.get("episodes", []) if e.get("id") != episode_id
        ]
        self._write_metadata(meta)

    # ── metadata ───────────────────────────────────────────────────────

    def _describe(self) -> dict:
        cfg = self._cfg
        manip = cfg["observation_manipulator"]
        return {
            "dataset_name": cfg["dataset_name"],
            "collect_hz": cfg["recording"]["collect_hz"],
            "robot": manip.get("name", ""),
            "gripper": cfg["observation_gripper"].get("name", ""),
            "joint_names": manip["joint"]["joint_names"],
            "cameras": {
                name: {
                    "resolution": cam["resolution"],
                    "rotate": cam.get("rotate", 0),
                }
                for name, cam in cfg["observation_cameras"].items()
            },
            "video_codec": cfg["storage"]["video_codec"],
            "features": self.feature_spec(),
        }

    def feature_spec(self) -> dict:
        """Shape/dtype/unit of every recorded stream, for consumers."""
        n_joints = len(
            self._cfg["observation_manipulator"]["joint"]["joint_names"]
        )
        spec = {
            "timestamp": {"shape": [1], "dtype": "float64", "unit": "s"},
            "stamp_ns": {"shape": [1], "dtype": "int64", "unit": "ns"},
            "time_left": {"shape": [1], "dtype": "float32", "unit": "s"},
            "observation.joint_position": {
                "shape": [n_joints], "dtype": "float32", "unit": "rad"},
            "observation.eef_position": {
                "shape": [3], "dtype": "float32", "unit": "m"},
            "observation.eef_orientation": {
                "shape": [4], "dtype": "float32", "unit": "quat(xyzw)"},
            "observation.eef_rot6d": {
                "shape": [6], "dtype": "float32",
                "unit": "rotmat cols 1,2 (column-major)"},
            "observation.eef_velocity": {
                "shape": [6], "dtype": "float32", "unit": "m/s,rad/s"},
            "observation.gripper_position": {
                "shape": [1], "dtype": "float32", "unit": "0..1"},
            "observation.gripper_effort": {
                "shape": [1], "dtype": "float32", "unit": "0..1"},
            "observation.gripper_effort_raw": {
                "shape": [1], "dtype": "float32", "unit": "raw"},
            "action.cartesian_velocity": {
                "shape": [6], "dtype": "float32", "unit": "m/s,rad/s"},
            "action.gripper": {
                "shape": [1], "dtype": "float32", "unit": "0..1"},
            "subtask_index": {"shape": [1], "dtype": "int32", "unit": "index"},
            "subtask_score": {"shape": [1], "dtype": "float32", "unit": "0..1"},
            "stale": {"shape": [1], "dtype": "uint8", "unit": "bitmask"},
        }
        if self._cfg["observation_manipulator"]["joint"].get("record_velocity", True):
            spec["observation.joint_velocity"] = {
                "shape": [n_joints], "dtype": "float32", "unit": "rad/s"}
        return spec

    def _write_metadata(self, meta: dict) -> None:
        merged = self._describe()
        merged["episodes"] = meta.get("episodes", [])
        merged["updated_at"] = datetime.now(timezone.utc).isoformat()
        tmp = self.metadata_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n")
        tmp.replace(self.metadata_path)
