#!/usr/bin/env python3
"""Convert every raw max_data_collect episode under `merge_target_dir` into one
LeRobot 3.0 dataset, driven by config/data_convert_lerobot_config.yaml.

Why this is not just `LeRobotDataset.add_frame()` in a loop:

    `add_frame` expects decoded image arrays for every video feature, so it
    would decode all 876 mp4s frame by frame and re-encode them. The recorder
    already wrote one mp4 per camera per episode, at the exact fps and
    resolution LeRobot wants. So instead we hand LeRobot the finished files and
    let its own concatenation step stream-copy them: minutes instead of hours.

    With `video.codec: copy` the pixels are bit-identical to the originals.
    With `video.codec: h264` each file is re-encoded once on the way through --
    still one encode per file rather than a decode/re-encode of every frame
    twice over -- because no major browser decodes HEVC in a <video> element,
    and an HEVC dataset therefore plays in neither the LeRobot visualizer nor
    on the Hub. Frame counts are verified to survive the re-encode.

Two details the raw data forces on us:

    * `timestamp` in the HDF5 is wall-clock and drifts from the ideal i/fps
      grid (median 0.6 ms, worst 18.5 ms -- every episode exceeds LeRobot's
      1e-4 s tolerance). LeRobot uses `timestamp` to look up video frames, so
      it must be i/fps or frame lookup breaks. The recorded clock is preserved
      separately as `observation.timestamp_raw`.
    * The editor (max_data_editor) stores its segment labels in the `edits.json`
      sidecar and never writes them back into the HDF5 -- the recorder's own
      `subtask_index`/`subtask_score` datasets are still -1/NaN everywhere. So
      the subtask columns are built from `edits.json`, not from the HDF5.
      Frames no segment covers keep index -1 and score -1.0 ("no subtask"),
      the -1.0 standing in for NaN so it cannot poison dataset statistics.

Subtasks land in three places, because two different consumers want two
different spellings of the same label:

    * `subtask_index` (per frame) + `meta/subtasks.parquet` -- the index looks
      up the full sentence prompt, which is what a VLA model is conditioned on.
      `LeRobotDataset.__getitem__` follows exactly this path to set `subtask`.
    * `sparse_subtask_*` columns on `meta/episodes/*.parquet` -- the short keys
      (`pick`, `place`, ...), matched by exact string against a global stage
      vocabulary by SARM, and drawn on the visualizer's subtask timeline.
    * `subtask_score` (per frame) -- the editor's segment quality score, which
      neither LeRobot schema has a field for.

Usage:
    python3 convert_to_lerobot.py --out /path/to/output            # convert
    python3 convert_to_lerobot.py --out ... --dry-run              # plan only
    python3 convert_to_lerobot.py --out ... --limit 5              # smoke test
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "data_convert_lerobot_config.yaml"


# ────────────────────────────────────────────────────────── config parsing
def _split_sources(value: str) -> list[str]:
    """`"a, b"` -> `["a", "b"]`; a YAML list passes through."""
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    root = Path(str(cfg.get("merge_target_dir", "")).strip())
    if not root.is_dir():
        raise SystemExit(f"merge_target_dir is not a directory: {root}")

    vectors: list[dict] = []          # numeric features built by concatenating arrays
    for group in ("action", "observation"):
        node = cfg.get(group) or {}
        blocks = {"action": {"": node}, "observation": {"state": node.get("state") or {}}}[group]
        for _, block in blocks.items():
            for _, spec in (block or {}).items():
                if not isinstance(spec, dict) or "source" not in spec:
                    continue
                names = spec.get("names")
                vectors.append({
                    "name": spec.get("rename") or spec["source"],
                    "sources": _split_sources(spec["source"]),
                    # Column labels, or None to leave the vector unlabelled.
                    "names": [str(n).strip() for n in names] if names else None,
                })

    cameras = {}
    for _, spec in ((cfg.get("observation") or {}).get("images") or {}).items():
        cameras[spec["rename"]] = str(spec["source"]).strip()

    # `metadata`/`subtask`: scalar per-frame columns. A value that names an
    # HDF5 dataset is read from it; a number is written as a constant.
    scalars: list[dict] = []
    for section, prefix in (("metadata", "observation"), ("subtask", "subtask")):
        for key, value in (cfg.get(section) or {}).items():
            name = f"{prefix}.{key}" if section == "metadata" else f"{prefix}_{key}"
            # The subtask columns come from the editor sidecar, not the HDF5
            # (see module docstring); the config value is only a column name.
            if section == "subtask":
                scalars.append({"name": name, "key": None, "const": None,
                                "from_edits": key})
                continue
            # LeRobot owns the top-level `timestamp` (it indexes video frames),
            # so the recorded wall clock is kept under a distinct name.
            if name == "observation.timestamp":
                name = "observation.timestamp_raw"
            entry = {"name": name, "key": None, "const": None, "from_edits": None}
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                entry["const"] = float(value)
            else:
                entry["key"] = str(value).strip()
            scalars.append(entry)

    video = dict(cfg.get("video") or {})
    video.setdefault("codec", "copy")
    if video["codec"] not in ("copy", "h264"):
        raise SystemExit(f"video.codec must be 'copy' or 'h264', not {video['codec']!r}")

    return {
        "root": root,
        "robot_type": (cfg.get("lerobot") or {}).get("robot_type", "unknown"),
        "vectors": vectors,
        "cameras": cameras,
        "scalars": scalars,
        "video": video,
    }


# ────────────────────────────────────────────────────────── dataset scanning
def discover_episodes(root: Path) -> list[Path]:
    """Every episode dir under root, ordered by dataset then episode number."""
    episodes = []
    for dataset in sorted(p for p in root.iterdir() if (p / "episodes").is_dir()):
        episodes.extend(sorted((dataset / "episodes").iterdir()))
    return [e for e in episodes if (e / "data.hdf5").is_file()]


def episode_prompt(ep: Path) -> str:
    """Prompt for an episode: the editor's edits.json wins over tasks.json."""
    prompt = ""
    for name in ("tasks.json", "edits.json"):
        path = ep / name
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8")).get("main_prompt")
            except ValueError:
                continue  # a corrupt sidecar must not lose the tasks.json prompt
            if value:
                prompt = value
    return prompt


def load_edits(ep: Path) -> dict:
    """The editor's sidecar for one episode, or `{}` when absent/unparsable."""
    path = ep / "edits.json"
    if not path.is_file():
        return {}
    try:
        edits = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    return edits if isinstance(edits, dict) else {}


def episode_score(ep: Path, default: float = 1.0) -> float:
    """The editor's episode-level quality score."""
    value = load_edits(ep).get("score")
    return float(value) if isinstance(value, (int, float)) else default


def episode_segments(ep: Path, start: int, end: int) -> list[dict]:
    """Editor segments clipped to the trim window, re-based to frame 0.

    The converter only writes frames `[start, end)`, so a segment's absolute
    frame numbers have to shift by `start` to line up with the output episode.
    Segments with no label are dropped: an unlabelled span is indistinguishable
    from one the annotator never reached, and both mean "no subtask here".
    """
    segments = []
    for seg in load_edits(ep).get("segments") or []:
        label = str(seg.get("label") or "").strip()
        prompt = str(seg.get("prompt") or "").strip()
        if not label:
            continue
        lo = max(int(seg.get("start", 0)), start)
        hi = min(int(seg.get("end", 0)), end)
        if hi <= lo:
            continue
        score = seg.get("score")
        segments.append({
            "label": label,
            "prompt": prompt or label,
            "start": lo - start,
            "end": hi - start,
            "score": float(score) if isinstance(score, (int, float)) else -1.0,
        })
    return sorted(segments, key=lambda s: s["start"])


def episode_trim(ep: Path, num_frames: int) -> tuple[int, int]:
    """Honour the editor's trim range when one was saved."""
    edits = load_edits(ep)
    if not edits:
        return 0, num_frames
    if edits.get("rejected"):
        return 0, 0
    trim = edits.get("trim") or {}
    start = max(0, min(int(trim.get("start", 0)), num_frames))
    end = max(start, min(int(trim.get("end", num_frames)), num_frames))
    return start, end


def build_features(cfg: dict, sample: Path, fps: float) -> dict:
    """Feature spec for LeRobotDataset.create, sized from a real episode."""
    features: dict[str, dict] = {}
    with h5py.File(sample / "data.hdf5", "r") as f:
        for vec in cfg["vectors"]:
            width = 0
            for key in vec["sources"]:
                if key not in f:
                    raise SystemExit(f"{sample}: HDF5 has no dataset '{key}'")
                width += f[key].shape[1] if f[key].ndim > 1 else 1
            names = vec["names"]
            # A name list that does not match the real width would label every
            # column after the discrepancy with its neighbour's name -- which a
            # policy then trains on. Refuse rather than mislabel.
            if names is not None and len(names) != width:
                raise SystemExit(
                    f"{vec['name']}: config lists {len(names)} names "
                    f"{names} but the data is {width} wide "
                    f"({', '.join(vec['sources'])})"
                )
            features[vec["name"]] = {"dtype": "float32", "shape": (width,), "names": names}
        for sc in cfg["scalars"]:
            if sc["key"] and sc["key"] not in f:
                raise SystemExit(f"{sample}: HDF5 has no dataset '{sc['key']}'")
            # `subtask_index` is a row number, not a measurement: LeRobot indexes
            # meta/subtasks.parquet with `.iloc[...]`, which rejects a float.
            # int64 also matches the `task_index` LeRobot defines for itself.
            dtype = "int64" if sc["name"] == "subtask_index" else "float32"
            features[sc["name"]] = {"dtype": dtype, "shape": (1,), "names": None}

    for name, filename in cfg["cameras"].items():
        width, height = _video_size(sample / filename)
        features[name] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def _video_size(path: Path) -> tuple[int, int]:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return stream.codec_context.width, stream.codec_context.height


# ────────────────────────────────────────────────────────── frame assembly
def subtask_columns(segments: list[dict], n: int, registry: dict[str, int]) -> dict:
    """Per-frame `subtask_index` / `subtask_score` for one episode.

    `registry` maps a sentence prompt to its row in `meta/subtasks.parquet`
    and is shared across the whole dataset, so it grows as new prompts appear.
    Frames outside every segment stay at -1 / -1.0 ("no subtask").
    """
    index = np.full((n, 1), -1, dtype=np.int64)
    score = np.full((n, 1), -1.0, dtype=np.float32)
    for seg in segments:
        idx = registry.setdefault(seg["prompt"], len(registry))
        index[seg["start"]:seg["end"]] = idx
        score[seg["start"]:seg["end"]] = seg["score"]
    return {"subtask_index": index, "subtask_score": score}


def read_episode(ep: Path, cfg: dict, fps: float, start: int, end: int,
                 registry: dict[str, int]) -> dict:
    """Numeric buffer for one episode, already trimmed to [start, end)."""
    n = end - start
    buf: dict = {}
    from_edits = subtask_columns(episode_segments(ep, start, end), n, registry)
    with h5py.File(ep / "data.hdf5", "r") as f:
        for vec in cfg["vectors"]:
            parts = [np.asarray(f[k][start:end]).reshape(n, -1) for k in vec["sources"]]
            buf[vec["name"]] = np.concatenate(parts, axis=1).astype(np.float32)
        for sc in cfg["scalars"]:
            if sc["from_edits"]:
                col = from_edits[f"subtask_{sc['from_edits']}"]
            elif sc["name"] == "observation.score":
                # The config carries a constant here, but the editor now has a
                # real per-episode score; prefer it, and fall back to the
                # constant for an episode with no sidecar.
                col = np.full((n, 1), episode_score(ep, sc["const"] or 1.0),
                              dtype=np.float32)
            elif sc["const"] is not None:
                col = np.full((n, 1), sc["const"], dtype=np.float32)
            else:
                col = np.asarray(f[sc["key"]][start:end]).reshape(n, 1).astype(np.float32)
                # NaN means "unlabelled"; keep it out of the dataset statistics.
                col = np.nan_to_num(col, nan=-1.0, posinf=-1.0, neginf=-1.0)
            buf[sc["name"]] = col
    return buf


def sample_frames(video: Path, num_frames: int) -> np.ndarray:
    """The frames LeRobot samples for image statistics, as a uint8 array.

    LeRobot's own helper takes file paths, which would mean writing ~200 PNGs
    per camera per episode. Sampling spans the whole clip either way, so the
    decode cost is unavoidable -- but the PNG round-trip is not.
    Returns (N, 3, H, W), matching what `sample_images` would have produced.
    """
    import av
    from lerobot.datasets.compute_stats import auto_downsample_height_width, sample_indices

    wanted = sorted({int(i) for i in sample_indices(num_frames)}) or [0]
    target, last = set(wanted), wanted[-1]
    frames = []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"          # use all cores for decoding
        for i, frame in enumerate(container.decode(stream)):
            if i in target:
                rgb = frame.to_ndarray(format="rgb24").transpose(2, 0, 1)
                frames.append(auto_downsample_height_width(rgb))
            if i >= last:
                break
    return np.stack(frames) if frames else np.zeros((1, 3, 1, 1), dtype=np.uint8)


# ────────────────────────────────────────────────────────────────── video
def _frame_count(path: Path) -> int:
    """Frames actually decodable from `path`.

    `nb_frames` from the container header is not trusted here: it is metadata,
    and the whole point of this check is to catch an encode that dropped a
    frame. Counting packets reads the file.
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_packets", "-show_entries", "stream=nb_read_packets",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return int(out) if out.isdigit() else 0


def _nvenc_available(encoder: str) -> bool:
    if not encoder.endswith("_nvenc"):
        return True
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return False
    return any(line.split()[1:2] == [encoder] for line in out.splitlines())


def transcode_h264(src: Path, dst: Path, cfg_video: dict) -> None:
    """Re-encode `src` to browser-playable H.264, frame count preserved.

    The recorder writes HEVC by default on some rigs, which no major browser
    decodes in a <video> element -- such a dataset plays in neither the LeRobot
    visualizer nor on the Hub. Re-encoding is a second generation of loss on
    top of the recorder's, which is why it is opt-in via `video.codec`.

    No `-r`, no `-vsync`, no filters: anything that resamples time would slide
    the video off the HDF5 rows, and every subtask boundary is a frame index
    into those rows. The result is counted and compared before it is accepted.
    """
    encoder = str(cfg_video.get("encoder", "h264_nvenc"))
    if not _nvenc_available(encoder):
        encoder = "libx264"

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
           "-c:v", encoder, "-pix_fmt", "yuv420p"]
    if encoder.endswith("_nvenc"):
        cmd += ["-cq", str(cfg_video.get("nvenc_cq", 26)),
                "-preset", str(cfg_video.get("nvenc_preset", "p4"))]
    else:
        cmd += ["-crf", str(cfg_video.get("x264_crf", 23)),
                "-preset", str(cfg_video.get("x264_preset", "veryfast"))]
    # `high` profile and a front-loaded moov atom are what browsers expect;
    # without faststart a player fetches the whole file before frame one.
    cmd += ["-profile:v", "high", "-movflags", "+faststart", "-an", str(dst)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"transcode failed for {src}:\n{result.stderr[:400]}")

    before, after = _frame_count(src), _frame_count(dst)
    if before != after:
        raise SystemExit(
            f"transcode changed the frame count for {src}: {before} -> {after}. "
            "Video and HDF5 rows would no longer line up, so the subtask "
            "boundaries would point at the wrong frames."
        )


# ─────────────────────────────────────────────────────── subtask metadata
def write_subtasks_parquet(out: Path, registry: dict[str, int]) -> None:
    """`meta/subtasks.parquet`: subtask_index -> sentence prompt.

    Mirrors `meta/tasks.parquet`, whose index holds the natural-language string
    (`load_tasks` sets `tasks.index.name = "task"`). This is the table
    `LeRobotDataset.__getitem__` reads to turn a frame's `subtask_index` into
    the `subtask` string. LeRobot 0.5.1 ships the reader (`load_subtasks`) but
    no writer, so the file is written here.
    """
    import pandas as pd

    if not registry:
        return
    prompts = [p for p, _ in sorted(registry.items(), key=lambda kv: kv[1])]
    # LeRobot looks a subtask up with `.iloc[subtask_index]`, so the -1 that
    # marks an unlabelled frame would wrap around to the LAST row and report a
    # real subtask. A trailing sentinel makes that wraparound land on an
    # explicit "no subtask" instead of a wrong one.
    prompts.append("")
    frame = pd.DataFrame(
        {"subtask_index": list(range(len(prompts) - 1)) + [-1]},
        index=pd.Index(prompts, name="subtask"),
    )
    path = out / "meta" / "subtasks.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow", compression="snappy")


def write_episode_annotations(out: Path, fps: float,
                              per_episode: dict[int, list[dict]]) -> None:
    """`sparse_subtask_*` columns on `meta/episodes/*.parquet`.

    The schema LeRobot's SARM annotator writes, and what the visualizer draws
    on its subtask timeline. Names are the editor's SHORT keys (`pick`), not
    the sentence prompts: SARM resolves a stage by exact string match against a
    global vocabulary, so a sentence with the object name substituted in would
    fragment that vocabulary and silently fall back to stage 0.
    """
    import pandas as pd

    from lerobot.datasets.utils import DEFAULT_EPISODES_PATH

    suffixes = ["names", "start_times", "end_times", "start_frames", "end_frames"]
    # `sparse_`-prefixed plus the unprefixed legacy spelling, both of which the
    # SARM loader accepts (it falls back from one to the other).
    columns = [f"sparse_subtask_{c}" for c in suffixes] + [f"subtask_{c}" for c in suffixes]

    for path in sorted((out / "meta" / "episodes").rglob("*.parquet")):
        frame = pd.read_parquet(path)
        if "episode_index" not in frame.columns:
            continue
        values = {col: [None] * len(frame) for col in columns}
        touched = False
        for row, ep_idx in enumerate(frame["episode_index"]):
            segments = per_episode.get(int(ep_idx))
            if not segments:
                continue
            touched = True
            cell = [
                [s["label"] for s in segments],
                [s["start"] / fps for s in segments],
                [s["end"] / fps for s in segments],
                [s["start"] for s in segments],
                [s["end"] for s in segments],
            ]
            for i, col in enumerate(columns):
                values[col][row] = cell[i % len(suffixes)]
        if not touched:
            continue
        for col in columns:
            frame[col] = pd.Series(values[col], index=frame.index, dtype=object)
        frame.to_parquet(path, engine="pyarrow", compression="snappy")


# ────────────────────────────────────────────────────────────────── convert
def convert(cfg: dict, out: Path, limit: int | None, overwrite: bool) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episodes = discover_episodes(cfg["root"])
    if limit:
        episodes = episodes[:limit]
    if not episodes:
        raise SystemExit(f"no episodes found under {cfg['root']}")

    fps = _dataset_fps(episodes[0])
    if out.exists():
        if not overwrite:
            raise SystemExit(f"output exists (use --overwrite): {out}")
        shutil.rmtree(out)

    features = build_features(cfg, episodes[0], fps)
    ds = LeRobotDataset.create(
        repo_id=f"local/{out.name}",
        fps=int(round(fps)),
        features=features,
        root=out,
        robot_type=cfg["robot_type"],
        use_videos=True,
    )

    injected: dict[str, Path] = {}
    registry: dict[str, int] = {}              # sentence prompt -> subtask_index
    annotations: dict[int, list[dict]] = {}    # episode_index -> editor segments

    def fake_worker(video_key, episode_index, root, fps_, vcodec, threads=None):
        # Stand in for png->mp4 encoding: hand over the recorder's own mp4,
        # re-encoded first if the config asks for it.
        # LeRobot rmtree()s this file's parent, so it needs a private dir.
        holder = Path(tempfile.mkdtemp(prefix="inject_"))
        dst = holder / f"{video_key}_{episode_index}.mp4"
        if cfg["video"]["codec"] == "h264":
            transcode_h264(injected[video_key], dst, cfg["video"])
        else:
            shutil.copy(injected[video_key], dst)
        return dst

    def fake_method(self, video_key, episode_index):
        return fake_worker(video_key, episode_index, self.root, self.fps, None)

    def passthrough_sample_images(data):
        # We hand the buffer real pixel arrays instead of PNG paths, so the
        # sampling/loading LeRobot would do here is already done.
        return np.asarray(data)

    written = skipped = 0
    for i, ep in enumerate(episodes, 1):
        with h5py.File(ep / "data.hdf5", "r") as f:
            total = int(f["timestamp"].shape[0])
        start, end = episode_trim(ep, total)
        n = end - start
        if n <= 0:
            skipped += 1
            print(f"[{i}/{len(episodes)}] skip {ep.parent.parent.name}/{ep.name} (rejected/empty)")
            continue

        buf = read_episode(ep, cfg, fps, start, end, registry)
        prompt = episode_prompt(ep)
        segments = episode_segments(ep, start, end)
        if segments:
            annotations[ds.meta.total_episodes] = segments
        buf.update({
            "size": n,
            "task": [prompt] * n,
            "episode_index": ds.meta.total_episodes,
            "frame_index": np.arange(n),
            # Regular grid, not the drifting wall clock -- see module docstring.
            "timestamp": (np.arange(n) / fps).astype(np.float32),
            "index": np.zeros(n, dtype=np.int64),
            "task_index": np.zeros(n, dtype=np.int64),
        })

        # Decode the statistics samples for all cameras concurrently.
        with ThreadPoolExecutor(max_workers=len(cfg["cameras"]) or 1) as pool:
            futures = {
                name: pool.submit(sample_frames, ep / filename, n)
                for name, filename in cfg["cameras"].items()
            }
            for name, filename in cfg["cameras"].items():
                injected[name] = ep / filename
                buf[name] = futures[name].result()

        with patch("lerobot.datasets.lerobot_dataset._encode_video_worker", fake_worker), \
             patch.object(LeRobotDataset, "_encode_temporary_episode_video", fake_method), \
             patch("lerobot.datasets.compute_stats.sample_images", passthrough_sample_images):
            ds.save_episode(episode_data=buf, parallel_encoding=False)

        written += 1
        trimmed = "" if (start, end) == (0, total) else f" trim[{start}:{end}]"
        print(f"[{i}/{len(episodes)}] {ep.parent.parent.name}/{ep.name}: {n} frames{trimmed}")

    ds.finalize()   # flush meta/episodes/, without which the dataset will not load

    # Both writers run after finalize(): they edit the metadata it just wrote.
    write_subtasks_parquet(out, registry)
    write_episode_annotations(out, fps, annotations)

    print(f"\nepisodes written : {written} (skipped {skipped})")
    print(f"frames           : {ds.meta.total_frames}")
    print(f"tasks            : {ds.meta.total_tasks}")
    print(f"subtasks         : {len(registry)} prompts over "
          f"{len(annotations)} annotated episodes")
    print(f"output           : {out}")


def _dataset_fps(ep: Path) -> float:
    meta = ep.parent.parent / "metadata.json"
    if meta.is_file():
        try:
            return float(json.loads(meta.read_text(encoding="utf-8"))["collect_hz"])
        except (ValueError, KeyError):
            pass
    return 30.0


def plan(cfg: dict, limit: int | None) -> None:
    episodes = discover_episodes(cfg["root"])
    if limit:
        episodes = episodes[:limit]
    frames = kept = rejected = 0
    prompts: dict[str, int] = {}
    labels: dict[str, int] = {}
    subtask_prompts: set[str] = set()
    unannotated = covered = 0
    for ep in episodes:
        with h5py.File(ep / "data.hdf5", "r") as f:
            total = int(f["timestamp"].shape[0])
        start, end = episode_trim(ep, total)
        if end - start <= 0:
            rejected += 1
            continue
        kept += 1
        frames += end - start
        prompts[episode_prompt(ep)] = prompts.get(episode_prompt(ep), 0) + 1
        segments = episode_segments(ep, start, end)
        if not segments:
            unannotated += 1
        for seg in segments:
            labels[seg["label"]] = labels.get(seg["label"], 0) + 1
            subtask_prompts.add(seg["prompt"])
            covered += seg["end"] - seg["start"]

    fps = _dataset_fps(episodes[0]) if episodes else 30.0
    print(f"root          : {cfg['root']}")
    print(f"robot_type    : {cfg['robot_type']}   fps: {fps}")
    print(f"episodes      : {kept} kept, {rejected} rejected/empty")
    print(f"frames        : {frames}  (~{frames / fps / 60:.1f} min)")
    # Same construction the real run uses, so a bad `names` length fails here
    # rather than an hour into the conversion.
    features = build_features(cfg, episodes[0], fps) if episodes else {}

    print("\nfeatures:")
    for vec in cfg["vectors"]:
        print(f"  {vec['name']:32s} <- {', '.join(vec['sources'])}")
        names = features.get(vec["name"], {}).get("names")
        width = features.get(vec["name"], {}).get("shape", (0,))[0]
        print(f"  {'':32s}    ({width}) {names if names else 'unnamed'}")
    for sc in cfg["scalars"]:
        if sc["from_edits"]:
            origin = f"edits.json segments.{sc['from_edits']}"
        elif sc["name"] == "observation.score":
            origin = f"edits.json score (default {sc['const']})"
        elif sc["const"] is not None:
            origin = f"const {sc['const']}"
        else:
            origin = sc["key"]
        print(f"  {sc['name']:32s} <- {origin}")
    note = ("stream-copy, no re-encode" if cfg["video"]["codec"] == "copy"
            else f"re-encode -> H.264 ({cfg['video'].get('encoder', 'h264_nvenc')})")
    for name, filename in cfg["cameras"].items():
        print(f"  {name:32s} <- {filename}  ({note})")
    print(f"\ntasks ({len(prompts)}):")
    for text, count in sorted(prompts.items(), key=lambda kv: -kv[1]):
        print(f"  {count:4d}  {text[:70]}")

    share = covered / frames * 100 if frames else 0.0
    print(f"\nsubtasks ({len(labels)} labels, {len(subtask_prompts)} prompts, "
          f"{share:.1f}% of frames covered, {unannotated} episodes unannotated):")
    for label, count in sorted(labels.items(), key=lambda kv: -kv[1]):
        print(f"  {count:4d}  {label}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out", type=Path, help="output dataset directory")
    ap.add_argument("--limit", type=int, help="convert only the first N episodes")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing output")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.dry_run:
        plan(cfg, args.limit)
        return
    if not args.out:
        ap.error("--out is required (or use --dry-run)")
    convert(cfg, args.out.resolve(), args.limit, args.overwrite)


if __name__ == "__main__":
    sys.exit(main())
