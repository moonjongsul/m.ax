"""FastAPI backend for the VLA dataset editor.

Settings come from config/editor_config.yaml; command-line flags override it.

Run:  ros2 run max_data_collect max_data_editor
      python3 -m max_data_collect.editor.server --config <editor_config.yaml>
"""
from __future__ import annotations

import argparse
import mimetypes
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

from . import config as editor_config
from .signals import read_series, suggest_segments
from .store import Catalog
from .transcode import get_proxy, sweep_stale

STATIC_DIR = Path(__file__).parent / "static"
CHUNK = 1024 * 512

app = FastAPI(title="M.AX Dataset Editor")
_catalog: Catalog | None = None
_cfg: dict[str, Any] = editor_config.DEFAULTS


def catalog() -> Catalog:
    if _catalog is None:  # pragma: no cover - configured at startup
        raise HTTPException(500, "catalog not configured")
    return _catalog


def cfg() -> dict[str, Any]:
    return _cfg


def configure(root: str | Path, config: dict[str, Any] | None = None) -> None:
    global _catalog, _cfg
    if config is not None:
        _cfg = config
    _catalog = Catalog(
        root,
        history_depth=_cfg["edits"]["history_depth"],
        default_vocabulary=_cfg["labels"]["default_vocabulary"],
    )


class RootIn(BaseModel):
    root: str


@app.get("/api/root")
def api_get_root() -> dict[str, Any]:
    cat = catalog()
    return {"root": str(cat.root), "exists": cat.root.is_dir(),
            "config_path": _cfg.get("_path")}


@app.put("/api/root")
def api_set_root(body: RootIn) -> dict[str, Any]:
    """Point the editor at a different dataset directory at runtime."""
    root = Path(body.root).expanduser()
    if not root.is_dir():
        raise HTTPException(400, f"not a directory: {root}")
    configure(root)
    return {"root": str(catalog().root), "datasets": catalog().list_datasets()}


# ------------------------------------------------------------------ schemas
class SegmentIn(BaseModel):
    start: int
    end: int
    label: str = ""
    prompt: str = ""
    score: float | None = None


class TrimIn(BaseModel):
    start: int
    end: int


class EditsIn(BaseModel):
    main_prompt: str | None = None
    notes: str | None = None
    score: float | None = None
    rejected: bool | None = None
    trim: TrimIn | None = None
    segments: list[SegmentIn] | None = None


class VocabIn(BaseModel):
    labels: list[str]


class BulkPromptIn(BaseModel):
    episodes: list[str]
    main_prompt: str


# --------------------------------------------------------------------- API
@app.get("/api/subtasks")
def api_subtasks() -> dict[str, Any]:
    """Prompt templates + slot vocabularies, straight from editor_config.yaml."""
    return {
        "subtasks": _cfg["subtasks"],
        "objects": _cfg["objects"],
        "targets": _cfg["targets"],
    }


@app.get("/api/datasets")
def api_datasets() -> dict[str, Any]:
    return {
        "root": str(catalog().root),
        "default_dataset": _cfg["dataset"]["default_dataset"],
        "datasets": catalog().list_datasets(),
    }


@app.get("/api/datasets/{dataset}")
def api_dataset(dataset: str) -> dict[str, Any]:
    try:
        return catalog().dataset_detail(dataset)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/datasets/{dataset}/episodes/{episode}")
def api_episode(dataset: str, episode: str,
                max_points: int | None = Query(None, ge=100, le=20000)) -> dict[str, Any]:
    cat = catalog()
    try:
        edir = cat.episode_dir(dataset, episode)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc

    meta = cat.dataset_detail(dataset)["metadata"]
    series: dict[str, Any] = {"num_frames": 0, "channels": [], "error": None}
    h5 = edir / "data.hdf5"
    if h5.is_file():
        try:
            series = read_series(
                h5,
                max_points=max_points or _cfg["timeline"]["max_points"],
                spec=_cfg["timeline"]["channels"],
                joint_names=meta.get("joint_names"),
            )
        except Exception as exc:  # h5py missing or file corrupt - keep the UI usable
            series["error"] = f"{type(exc).__name__}: {exc}"

    return {
        "dataset": dataset,
        "episode": episode,
        "fps": meta.get("collect_hz") or 30.0,
        "cameras": sorted(p.stem for p in edir.glob("*.mp4")),
        "keyframes": sorted(p.name for p in edir.glob("*.jpg")),
        "series": series,
        "edits": cat.get_edits(dataset, episode),
        "label_vocabulary": cat.load_vocabulary(dataset),
    }


@app.put("/api/datasets/{dataset}/episodes/{episode}/edits")
def api_save_edits(dataset: str, episode: str, body: EditsIn) -> dict[str, Any]:
    payload = body.model_dump(exclude_none=True)
    if body.segments is not None:
        payload["segments"] = [s.model_dump() for s in body.segments]
    if body.trim is not None:
        payload["trim"] = body.trim.model_dump()
    try:
        return catalog().save_edits(dataset, episode, payload)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/datasets/{dataset}/episodes/{episode}/suggest")
def api_suggest(dataset: str, episode: str) -> dict[str, Any]:
    cat = catalog()
    h5 = cat.episode_dir(dataset, episode) / "data.hdf5"
    if not h5.is_file():
        raise HTTPException(404, "data.hdf5 not found")
    # The gripper-edge offsets are configured in seconds, so the suggester
    # needs the capture rate to turn them into frames.
    fps = cat.dataset_detail(dataset)["metadata"].get("collect_hz") or 30.0
    try:
        return suggest_segments(h5, _cfg["autosegment"], fps=fps)
    except Exception as exc:
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc


@app.put("/api/datasets/{dataset}/vocabulary")
def api_vocab(dataset: str, body: VocabIn) -> dict[str, Any]:
    return {"labels": catalog().save_vocabulary(dataset, body.labels)}


@app.post("/api/datasets/{dataset}/bulk_prompt")
def api_bulk_prompt(dataset: str, body: BulkPromptIn) -> dict[str, Any]:
    cat = catalog()
    updated = []
    for ep in body.episodes:
        try:
            cat.save_edits(dataset, ep, {"main_prompt": body.main_prompt})
            updated.append(ep)
        except FileNotFoundError:
            continue
    return {"updated": updated}


# ------------------------------------------------------------------- media
@app.get("/api/datasets/{dataset}/episodes/{episode}/file/{filename}")
def api_file(dataset: str, episode: str, filename: str):
    """Serve jpg keyframes directly (small, no Range needed)."""
    path = _safe_media(dataset, episode, filename)
    return FileResponse(path)


def _safe_media(dataset: str, episode: str, filename: str) -> Path:
    if "/" in filename or ".." in filename:
        raise HTTPException(400, "bad filename")
    try:
        path = (catalog().episode_dir(dataset, episode) / filename).resolve()
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    if not path.is_file():
        raise HTTPException(404, f"no such file: {filename}")
    return path


@app.get("/api/datasets/{dataset}/episodes/{episode}/video/{filename}")
def api_video(dataset: str, episode: str, filename: str, range: str | None = None):
    """HTTP Range streaming so the browser can seek within the mp4."""
    path = _safe_media(dataset, episode, filename)
    # HEVC recordings are typically not playable in-browser; hand back a
    # cached H.264 proxy unless video.proxy says otherwise.
    video_cfg = _cfg["video"]
    path = get_proxy(path, catalog().dataset_dir(dataset) / video_cfg["cache_dirname"], video_cfg)
    size = path.stat().st_size
    ctype = mimetypes.guess_type(path.name)[0] or "video/mp4"

    if not range:
        return FileResponse(path, media_type=ctype)

    match = re.match(r"bytes=(\d*)-(\d*)", range)
    if not match:
        raise HTTPException(416, "bad range")
    start = int(match.group(1)) if match.group(1) else 0
    end = int(match.group(2)) if match.group(2) else size - 1
    start = max(0, min(start, size - 1))
    end = max(start, min(end, size - 1))
    length = end - start + 1

    def stream():
        with path.open("rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                data = fh.read(min(CHUNK, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    return StreamingResponse(
        stream(),
        status_code=206,
        media_type=ctype,
        headers={
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
        },
    )


# --------------------------------------------------------------------- app
@app.get("/", response_class=HTMLResponse)
def index() -> Response:
    page = STATIC_DIR / "index.html"
    if not page.is_file():
        raise HTTPException(500, "static/index.html missing")
    return HTMLResponse(page.read_text(encoding="utf-8"))


@app.get("/app.js")
def appjs() -> Response:
    path = STATIC_DIR / "app.js"
    if not path.is_file():
        raise HTTPException(404, "app.js missing")
    return Response(path.read_text(encoding="utf-8"), media_type="application/javascript")


def main() -> None:
    parser = argparse.ArgumentParser(description="M.AX dataset editor")
    parser.add_argument("--config", default=None,
                        help=f"editor config YAML (default: {editor_config.DEFAULT_CONFIG_PATH})")
    # These override the corresponding config values when given.
    parser.add_argument("--root", default=None, help="dataset root directory")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    try:
        conf = editor_config.load(args.config)
    except (RuntimeError, OSError) as exc:
        raise SystemExit(f"[max_data_editor] config error: {exc}") from exc

    if args.root:
        conf["dataset"]["root"] = args.root
    root = editor_config.resolve_root(conf)
    if not root.is_dir():
        raise SystemExit(f"[max_data_editor] dataset root not found: {root}")

    host = args.host or conf["server"]["host"]
    port = args.port or conf["server"]["port"]
    configure(root, conf)

    # A previous run killed mid-encode leaves .part files behind.
    stale = sum(sweep_stale(d / conf["video"]["cache_dirname"])
                for d in root.iterdir() if d.is_dir())
    if stale:
        print(f"[max_data_editor] removed {stale} stale proxy file(s)")

    import uvicorn

    print(f"[max_data_editor] config  : {conf.get('_path') or '(defaults)'}")
    print(f"[max_data_editor] root    : {root}")
    print(f"[max_data_editor] proxy   : {conf['video']['proxy']}")
    print(f"[max_data_editor] serving at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
