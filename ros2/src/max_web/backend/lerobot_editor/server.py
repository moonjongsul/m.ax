"""FastAPI backend for the LeRobot Editor page.

Reads LeRobot v3.0 datasets (codebase_version "v3.0") from a local path and
exposes per-episode metadata, action/state arrays, and the underlying mp4
files via HTTP range requests so the React frontend can drive a synchronized
video + plot timeline.

Run:
    uvicorn server:app --host 0.0.0.0 --port 8765 --reload
or:
    python server.py
"""
from __future__ import annotations

import copy
import json
import logging
import os
import threading
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

log = logging.getLogger("lerobot_editor")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="LeRobot Editor Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- in-memory cache of currently loaded dataset --------------------------

class DatasetState:
    path: Path | None = None
    info: dict[str, Any] | None = None
    episodes: list[dict[str, Any]] | None = None  # parsed episode rows
    episodes_by_index: dict[int, dict[str, Any]] | None = None


STATE = DatasetState()


# ---- helpers --------------------------------------------------------------

def _format_path(template: str, **kwargs: Any) -> str:
    """Render lerobot info.json path templates like 'data/chunk-{chunk_index:03d}/...'.

    Python str.format already supports the format specs used by lerobot
    (':03d', etc.), so we just delegate.
    """
    return template.format(**kwargs)


def _read_episodes_meta(root: Path) -> list[dict[str, Any]]:
    """Concatenate every parquet under meta/episodes/ and return a list of
    plain-dict rows (one per episode)."""
    ep_dir = root / "meta" / "episodes"
    if not ep_dir.is_dir():
        raise HTTPException(400, f"meta/episodes not found under {root}")

    files: list[Path] = sorted(p for p in ep_dir.rglob("*.parquet"))
    if not files:
        raise HTTPException(400, f"no episode parquet files under {ep_dir}")

    rows: list[dict[str, Any]] = []
    for f in files:
        table = pq.read_table(f)
        # We only need a subset of columns; pyarrow → python dicts is fine
        # because there are O(hundreds) of episodes.
        cols = table.column_names
        keep = [
            c for c in cols
            if c in {"episode_index", "tasks", "length",
                     "data/chunk_index", "data/file_index",
                     "dataset_from_index", "dataset_to_index"}
            or c.startswith("videos/")
        ]
        sub = table.select(keep)
        for i in range(sub.num_rows):
            row = {c: sub.column(c)[i].as_py() for c in keep}
            rows.append(row)

    rows.sort(key=lambda r: r["episode_index"])
    return rows


def _video_keys(info: dict[str, Any]) -> list[str]:
    """Return the list of feature names whose dtype is 'video'."""
    return [k for k, v in info.get("features", {}).items()
            if isinstance(v, dict) and v.get("dtype") == "video"]


def _require_loaded() -> DatasetState:
    if STATE.path is None or STATE.info is None or STATE.episodes is None:
        raise HTTPException(400, "no dataset loaded; POST /api/dataset/load first")
    return STATE


# ---- request/response models ---------------------------------------------

class LoadRequest(BaseModel):
    path: str


# ---- endpoints ------------------------------------------------------------

@app.post("/api/dataset/load")
def load_dataset(req: LoadRequest):
    root = Path(req.path).expanduser().resolve()
    if not root.is_dir():
        raise HTTPException(400, f"path is not a directory: {root}")

    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise HTTPException(400, f"meta/info.json not found at {info_path}")

    with info_path.open() as fh:
        info = json.load(fh)

    if info.get("codebase_version") != "v3.0":
        # Don't hard-fail; warn via response so the UI can show it.
        pass

    episodes = _read_episodes_meta(root)

    STATE.path = root
    STATE.info = info
    STATE.episodes = episodes
    STATE.episodes_by_index = {int(e["episode_index"]): e for e in episodes}

    return {
        "path": str(root),
        "codebase_version": info.get("codebase_version"),
        "robot_type": info.get("robot_type"),
        "fps": info.get("fps"),
        "total_episodes": info.get("total_episodes", len(episodes)),
        "total_frames": info.get("total_frames"),
        "video_keys": _video_keys(info),
        "action_names": info.get("features", {}).get("action", {}).get("names", []),
        "state_names": info.get("features", {}).get("observation.state", {}).get("names", []),
    }


@app.get("/api/dataset/episodes")
def list_episodes():
    s = _require_loaded()
    fps = s.info.get("fps", 30)
    out = []
    for e in s.episodes:
        tasks = e.get("tasks") or []
        if not isinstance(tasks, list):
            tasks = [str(tasks)]
        out.append({
            "episode_index": int(e["episode_index"]),
            "length": int(e["length"]),
            "duration": float(e["length"]) / float(fps),
            "tasks": tasks,
        })
    return {"fps": fps, "episodes": out}


@app.get("/api/dataset/episodes/{episode_index}/frames")
def episode_frames(episode_index: int):
    """Return per-frame action and state arrays for the given episode."""
    s = _require_loaded()
    if episode_index not in s.episodes_by_index:
        raise HTTPException(404, f"episode {episode_index} not found")
    ep = s.episodes_by_index[episode_index]

    chunk_idx = int(ep["data/chunk_index"])
    file_idx = int(ep["data/file_index"])
    row_from = int(ep["dataset_from_index"])
    row_to = int(ep["dataset_to_index"])

    rel = _format_path(s.info["data_path"], chunk_index=chunk_idx, file_index=file_idx)
    parquet_path = s.path / rel
    if not parquet_path.is_file():
        raise HTTPException(500, f"parquet missing: {parquet_path}")

    # Read only the columns we need, then slice by row index.
    cols = ["action", "observation.state", "timestamp", "frame_index"]
    table = pq.read_table(parquet_path, columns=[c for c in cols
                                                 if c in pq.ParquetFile(parquet_path).schema_arrow.names])
    table = table.slice(row_from, row_to - row_from)

    def col_as_list(name: str) -> list[Any] | None:
        if name not in table.column_names:
            return None
        return table.column(name).to_pylist()

    return {
        "episode_index": episode_index,
        "length": row_to - row_from,
        "fps": s.info.get("fps", 30),
        "action": col_as_list("action"),
        "state": col_as_list("observation.state"),
        "timestamp": col_as_list("timestamp"),
        "tasks": ep.get("tasks") or [],
        "videos": _video_clip_meta_for(ep, s.info),
    }


def _video_clip_meta_for(ep: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    """Per video_key: relative URL to mp4 + from/to timestamps inside it."""
    out: dict[str, Any] = {}
    for vk in _video_keys(info):
        prefix = f"videos/{vk}/"
        ck = int(ep[f"{prefix}chunk_index"])
        fk = int(ep[f"{prefix}file_index"])
        t0 = float(ep[f"{prefix}from_timestamp"])
        t1 = float(ep[f"{prefix}to_timestamp"])
        out[vk] = {
            "url": f"/api/dataset/video?video_key={vk}&chunk={ck}&file={fk}",
            "from_timestamp": t0,
            "to_timestamp": t1,
        }
    return out


@app.get("/api/dataset/video")
def get_video(video_key: str, chunk: int, file: int, request: Request):
    """Serve an mp4 with HTTP Range support so <video> can seek."""
    s = _require_loaded()
    rel = _format_path(s.info["video_path"],
                       video_key=video_key, chunk_index=chunk, file_index=file)
    path = s.path / rel
    if not path.is_file():
        raise HTTPException(404, f"video not found: {rel}")

    file_size = path.stat().st_size
    range_header = request.headers.get("range")
    if range_header is None:
        return FileResponse(path, media_type="video/mp4")

    # Parse "bytes=START-END"
    try:
        units, rng = range_header.split("=", 1)
        if units.strip() != "bytes":
            raise ValueError
        start_s, end_s = rng.split("-", 1)
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else file_size - 1
    except ValueError:
        raise HTTPException(416, "invalid Range header")

    end = min(end, file_size - 1)
    if start > end:
        raise HTTPException(416, "Range not satisfiable")

    length = end - start + 1
    with path.open("rb") as fh:
        fh.seek(start)
        data = fh.read(length)

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
        "Content-Type": "video/mp4",
    }
    return Response(content=data, status_code=206, headers=headers)


# ---- export ---------------------------------------------------------------

EXPORT_STATE: dict[str, Any] = {
    "running": False,
    "progress": 0.0,
    "current_episode": None,
    "total_episodes": 0,
    "message": "",
    "error": None,
    "out_path": None,
}
_EXPORT_LOCK = threading.Lock()


class ExportRequest(BaseModel):
    out_path: str
    # Per-episode trim ranges in episode-relative seconds, keyed by episode_index
    # as a string (JSON object key). Episodes not in this map are exported in full.
    trims: dict[str, dict[str, float]] = {}


def _load_tasks_map(root: Path) -> dict[int, str]:
    """Return task_index -> task string from meta/tasks.parquet."""
    p = root / "meta" / "tasks.parquet"
    if not p.is_file():
        raise HTTPException(500, f"tasks.parquet missing: {p}")
    t = pq.read_table(p)
    out: dict[int, str] = {}
    for i in range(t.num_rows):
        out[int(t.column("task_index")[i].as_py())] = str(t.column("task")[i].as_py())
    return out


def _user_features_from_info(info: dict[str, Any]) -> dict[str, Any]:
    """Strip auto-managed default features so they match what LeRobotDataset.create
    expects from the caller. Also coerce shape to tuple so validate_frame's
    `actual_shape != expected_shape` comparison works against numpy arrays."""
    auto = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    out: dict[str, Any] = {}
    for k, v in info["features"].items():
        if k in auto:
            continue
        v = copy.deepcopy(v)
        if isinstance(v.get("shape"), list):
            v["shape"] = tuple(v["shape"])
        out[k] = v
    return out


def _do_export(src_root: Path, src_info: dict[str, Any],
               episodes: list[dict[str, Any]], out_path: Path,
               trims: dict[int, tuple[float, float]]) -> None:
    """Worker that writes a trimmed copy of the dataset.

    Runs on a background thread so the HTTP request can return immediately and
    the frontend can poll /api/dataset/export/status.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import decode_video_frames

    fps = int(src_info["fps"])
    tol = 0.5 / fps  # half a frame
    features = _user_features_from_info(src_info)
    tasks_map = _load_tasks_map(src_root)

    with _EXPORT_LOCK:
        if EXPORT_STATE["running"]:
            raise RuntimeError("export already running")
        EXPORT_STATE.update({
            "running": True, "progress": 0.0, "current_episode": None,
            "total_episodes": len(episodes), "message": "starting…",
            "error": None, "out_path": str(out_path),
        })

    try:
        if out_path.exists():
            if any(out_path.iterdir()):
                raise RuntimeError(f"export path is not empty: {out_path}")
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        # Don't create out_path itself — LeRobotDataset.create handles it.

        log.info("creating LeRobotDataset at %s", out_path)
        ds = LeRobotDataset.create(
            repo_id="trimmed_export",
            fps=fps,
            features=features,
            root=out_path,
            robot_type=src_info.get("robot_type"),
            use_videos=True,
        )

        video_keys = _video_keys(src_info)

        for n_done, ep in enumerate(episodes):
            ep_idx = int(ep["episode_index"])
            t0_in, t1_in = trims.get(ep_idx, (0.0, ep["length"] / fps))
            length = int(ep["length"])
            full_dur = length / fps
            t0 = max(0.0, min(t0_in, full_dur))
            t1 = max(0.0, min(t1_in, full_dur))
            if t1 <= t0 + 0.5 / fps:
                log.info("skipping ep %d (empty trim)", ep_idx)
                EXPORT_STATE["current_episode"] = ep_idx
                EXPORT_STATE["progress"] = (n_done + 1) / len(episodes)
                continue

            f_start = int(round(t0 * fps))
            f_end = int(round(t1 * fps))
            f_end = min(f_end, length)
            n_frames = f_end - f_start
            if n_frames <= 0:
                continue

            EXPORT_STATE.update({
                "current_episode": ep_idx,
                "message": f"episode {ep_idx}: {n_frames} frames",
            })
            log.info("ep %d: frames [%d, %d) of %d", ep_idx, f_start, f_end, length)

            # --- read source action / state / task_index for the trimmed range
            chunk_idx = int(ep["data/chunk_index"])
            file_idx = int(ep["data/file_index"])
            row_from = int(ep["dataset_from_index"]) + f_start
            row_to = int(ep["dataset_from_index"]) + f_end
            parquet_path = src_root / _format_path(
                src_info["data_path"], chunk_index=chunk_idx, file_index=file_idx)
            tbl = pq.read_table(parquet_path,
                                columns=["action", "observation.state", "task_index"])
            tbl = tbl.slice(row_from - int(ep["dataset_from_index"]),
                            row_to - row_from)
            actions = tbl.column("action").to_pylist()
            states = tbl.column("observation.state").to_pylist()
            task_indices = tbl.column("task_index").to_pylist()

            # --- decode video frames for the trimmed range, per video key
            videos: dict[str, np.ndarray] = {}
            for vk in video_keys:
                vmeta_prefix = f"videos/{vk}/"
                vck = int(ep[f"{vmeta_prefix}chunk_index"])
                vfk = int(ep[f"{vmeta_prefix}file_index"])
                v_t0 = float(ep[f"{vmeta_prefix}from_timestamp"])
                vpath = src_root / _format_path(
                    src_info["video_path"],
                    video_key=vk, chunk_index=vck, file_index=vfk)
                # Episode timestamps inside the video file
                ts = [v_t0 + (f_start + i) / fps for i in range(n_frames)]
                frames_t = decode_video_frames(vpath, ts, tol)
                # torch.Tensor shape [N, C, H, W] uint8 in [0,255] or float [0,1]
                arr = frames_t.detach().cpu().numpy()
                if arr.dtype != np.uint8:
                    arr = (arr * 255).clip(0, 255).astype(np.uint8)
                # Convert to NHWC for lerobot's add_frame.
                if arr.ndim == 4 and arr.shape[1] in (1, 3) and arr.shape[-1] not in (1, 3):
                    arr = np.transpose(arr, (0, 2, 3, 1))
                videos[vk] = arr

            # --- emit frames
            for i in range(n_frames):
                frame: dict[str, Any] = {
                    "action": np.asarray(actions[i], dtype=np.float32),
                    "observation.state": np.asarray(states[i], dtype=np.float32),
                    "task": tasks_map[int(task_indices[i])],
                }
                for vk in video_keys:
                    frame[vk] = videos[vk][i]
                ds.add_frame(frame)
            ds.save_episode()

            EXPORT_STATE["progress"] = (n_done + 1) / len(episodes)

        EXPORT_STATE["message"] = "finalizing…"
        ds.finalize()
        EXPORT_STATE.update({
            "message": f"done → {out_path}",
            "progress": 1.0,
        })
        log.info("export finished: %s", out_path)
    except Exception as e:  # noqa: BLE001
        log.error("export failed: %s\n%s", e, traceback.format_exc())
        EXPORT_STATE["error"] = f"{type(e).__name__}: {e}"
        EXPORT_STATE["message"] = "failed"
    finally:
        EXPORT_STATE["running"] = False


@app.post("/api/dataset/export")
def export_dataset(req: ExportRequest):
    s = _require_loaded()
    if EXPORT_STATE["running"]:
        raise HTTPException(409, "another export is already running")
    out_path = Path(req.out_path).expanduser().resolve()
    if out_path == s.path:
        raise HTTPException(400, "out_path must differ from the source dataset path")

    trims_int: dict[int, tuple[float, float]] = {}
    for k, v in req.trims.items():
        try:
            ek = int(k)
        except ValueError:
            continue
        ts = float(v.get("trimStart", 0.0))
        te = float(v.get("trimEnd", 0.0))
        trims_int[ek] = (ts, te)

    # Snapshot dataset state for the worker (avoid races with re-loads).
    src_root = s.path
    src_info = copy.deepcopy(s.info)
    episodes = copy.deepcopy(s.episodes)

    th = threading.Thread(
        target=_do_export,
        args=(src_root, src_info, episodes, out_path, trims_int),
        daemon=True,
    )
    th.start()
    return {"started": True, "out_path": str(out_path)}


@app.get("/api/dataset/export/status")
def export_status():
    return dict(EXPORT_STATE)


@app.get("/api/health")
def health():
    return {"ok": True, "loaded": STATE.path is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
