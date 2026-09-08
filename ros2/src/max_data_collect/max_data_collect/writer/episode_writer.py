"""Background episode writer: HDF5 states plus one mp4 per camera.

Frames go through a bounded queue so disk and encoder latency never stall
the 30 Hz sampling loop. If the queue fills, that is a real failure and
the recorder says so (`recording.on_overflow`) rather than quietly losing
the middle of a demonstration.

States are buffered in memory for the whole episode and written in one
pass at the end. That is what lets `time_left` exist: it is
`duration - timestamp`, and the duration is not known until the operator
stops. Video is streamed frame by frame as it arrives, because buffering
raw frames would cost gigabytes.
"""

import json
import queue
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from max_data_collect.writer.keyframe_writer import KeyframeWriter
from max_data_collect.writer.video_writer import VideoWriter


class QueueOverflow(RuntimeError):
    """Raised into the sampling loop when the frame queue fills up."""


# Column -> (dtype, width). Width 1 means a flat (N,) array.
def _columns(n_joints: int, record_joint_vel: bool) -> dict:
    cols = {
        "timestamp": ("f8", 1),
        "stamp_ns": ("i8", 1),
        "observation.joint_position": ("f4", n_joints),
        "observation.eef_position": ("f4", 3),
        "observation.eef_orientation": ("f4", 4),
        "observation.eef_rot6d": ("f4", 6),
        "observation.eef_velocity": ("f4", 6),
        "observation.gripper_position": ("f4", 1),
        "observation.gripper_effort": ("f4", 1),
        "observation.gripper_effort_raw": ("f4", 1),
        "action.cartesian_velocity": ("f4", 6),
        "action.gripper": ("f4", 1),
        "stale": ("u1", 1),
    }
    if record_joint_vel:
        cols["observation.joint_velocity"] = ("f4", n_joints)
    return cols


class EpisodeWriter:
    """Owns one episode's queue, worker thread, and output files."""

    def __init__(self, logger, cfg: dict, episode_dir: Path, episode_id: str,
                 episode_index: int, stale_bits: dict):
        self._log = logger
        self._cfg = cfg
        self._dir = episode_dir
        self._episode_id = episode_id
        self._episode_index = episode_index
        self._stale_bits = stale_bits

        recording = cfg["recording"]
        self._queue: queue.Queue = queue.Queue(
            maxsize=int(recording["queue_size"])
        )
        self._on_overflow = recording["on_overflow"]
        self._hz = float(recording["collect_hz"])

        manip = cfg["observation_manipulator"]
        self._record_joint_vel = bool(manip["joint"].get("record_velocity", True))
        self._columns = _columns(
            len(manip["joint"]["joint_names"]), self._record_joint_vel
        )

        self._cameras = cfg["observation_cameras"]
        self._videos: dict[str, VideoWriter] = {}
        self._keyframes: KeyframeWriter | None = None
        self._rows: list[dict] = []
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._dropped = 0

    # ── lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        storage = self._cfg["storage"]
        for name, cam in self._cameras.items():
            width, height = cam["resolution"]
            if width <= 0 or height <= 0:
                raise RuntimeError(
                    f"camera '{name}': resolution must be set to encode video, "
                    f"got {cam['resolution']}"
                )
            self._videos[name] = VideoWriter(
                self._log, self._dir / f"{name}.mp4", width, height, self._hz,
                storage["video_codec"], int(storage["video_cq"]),
                storage["video_preset"],
            )
        keyframes = storage["keyframes"]
        if keyframes["enabled"]:
            self._keyframes = KeyframeWriter(
                self._log, self._dir, list(self._cameras),
                keyframes["composite_order"], keyframes["jpeg_quality"],
            )
        self._thread = threading.Thread(
            target=self._run, daemon=True,
            name=f"max-collect-writer-{self._episode_id}",
        )
        self._thread.start()

    def put(self, row: dict) -> None:
        """Enqueue a row. Never blocks the sampling loop."""
        if self._error is not None:
            raise RuntimeError(f"writer failed: {self._error}") from self._error
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            if self._on_overflow == "drop":
                self._dropped += 1
                return
            raise QueueOverflow(
                f"frame queue full ({self._queue.maxsize} frames); the writer "
                "cannot keep up with the sampling rate"
            )

    def finish(self, duration: float, timeout: float = 600.0) -> dict:
        """Drain, close the encoders, write HDF5 and tasks.json."""
        self._drain(timeout)
        if self._error is not None:
            raise RuntimeError(f"writer failed: {self._error}") from self._error

        counts = {name: v.close() for name, v in self._videos.items()}
        self._videos.clear()

        n = len(self._rows)
        mismatched = {k: c for k, c in counts.items() if c != n}
        if mismatched:
            # Every mp4 must be exactly N frames -- that invariant is why
            # no per-camera frame index is stored.
            raise RuntimeError(
                f"video frame count mismatch (expected {n}): {mismatched}"
            )

        keyframes = (
            self._keyframes.write() if self._keyframes is not None else []
        )
        self._write_hdf5(n, duration)
        self._write_tasks(n, keyframes)
        return {
            "id": self._episode_id,
            "index": self._episode_index,
            "num_frames": n,
            "duration": duration,
            "source": self._cfg["source"],
            "dropped": self._dropped,
        }

    def discard(self) -> None:
        """Abandon the episode and remove anything already on disk."""
        self._drain(30.0)
        for video in self._videos.values():
            video.abort()
        self._videos.clear()
        self._rows.clear()
        shutil.rmtree(self._dir, ignore_errors=True)

    def _drain(self, timeout: float) -> None:
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise TimeoutError(f"writer did not drain within {timeout}s")
            self._thread = None

    # ── worker ─────────────────────────────────────────────────────────

    def _run(self) -> None:
        while True:
            row = self._queue.get()
            if row is None:
                return
            try:
                images = row.pop("images")
                if self._keyframes is not None:
                    self._keyframes.observe(images)
                for name, video in self._videos.items():
                    video.write(images.get(name))
                self._rows.append(row)
            except BaseException as exc:            # noqa: BLE001
                self._error = exc
                self._log.error(f"[collect] writer failed: {exc}")
                return

    # ── output ─────────────────────────────────────────────────────────

    def _write_hdf5(self, n: int, duration: float) -> None:
        path = self._dir / "data.hdf5"
        with h5py.File(path, "w") as f:
            meta = f.create_group("meta")
            manip = self._cfg["observation_manipulator"]
            for key, value in {
                "episode_id": self._episode_id,
                "episode_index": self._episode_index,
                "source": self._cfg["source"],
                "collect_hz": self._hz,
                "num_frames": n,
                "duration": duration,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "robot": manip.get("name", ""),
                "gripper": self._cfg["observation_gripper"].get("name", ""),
                "segment_sec": self._cfg["task"]["segment_sec"],
                "segment_frames": self._cfg["task"]["segment_frames"],
                "joint_names": json.dumps(manip["joint"]["joint_names"]),
                "cameras": json.dumps(list(self._cameras)),
                "stale_bits": json.dumps(self._stale_bits),
            }.items():
                meta.attrs[key] = value

            for column, (dtype, width) in self._columns.items():
                shape = (n,) if width == 1 else (n, width)
                data = np.zeros(shape, dtype=dtype)
                for i, row in enumerate(self._rows):
                    data[i] = row[column]
                f.create_dataset(column, data=data, compression="gzip",
                                 compression_opts=4)

            # Counted backwards from the end, so frame 0 of a 40 s episode
            # reads 40.0 and the last frame reads 0.0. A trim tool must
            # recompute this.
            timestamps = np.array([r["timestamp"] for r in self._rows], dtype="f8")
            f.create_dataset(
                "time_left",
                data=np.maximum(0.0, duration - timestamps).astype("f4"),
                compression="gzip", compression_opts=4,
            )

            # Filled in by the editing GUI: -1 means "not yet labelled".
            f.create_dataset("subtask_index", data=np.full(n, -1, dtype="i4"),
                             compression="gzip", compression_opts=4)
            f.create_dataset("subtask_score", data=np.full(n, np.nan, dtype="f4"),
                             compression="gzip", compression_opts=4)

    def _write_tasks(self, n: int, keyframes: list[str]) -> None:
        task = self._cfg["task"]
        segment_frames = int(task["segment_frames"])
        num_segments = max(0, n - segment_frames + 1)
        if num_segments == 0:
            self._log.warning(
                f"[collect] {self._episode_id} has {n} frames, shorter than one "
                f"{segment_frames}-frame segment; it has no labellable segments"
            )
        payload = {
            "episode_id": self._episode_id,
            "episode_index": self._episode_index,
            "source": self._cfg["source"],
            "main_prompt": task["main_prompt"],
            "goal_image": task["goal_image"],
            "total_score": task["total_score"],
            "segment_sec": task["segment_sec"],
            "segment_frames": segment_frames,
            "num_frames": n,
            "num_segments": num_segments,
            # Filenames relative to this episode directory. The start
            # frames are frame 0 of each mp4 and the end frames are frame
            # N-1, so either set can be used as an init/goal conditioning
            # image without re-deriving it from the video.
            "keyframes": keyframes,
            # Unique subgoal strings, indexed by `subtask_index` in the
            # HDF5. Filled in by the editing GUI.
            "subtasks": [],
        }
        (self._dir / "tasks.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        )

    # ── introspection ──────────────────────────────────────────────────

    @property
    def written(self) -> int:
        return len(self._rows)

    @property
    def dropped(self) -> int:
        return self._dropped
