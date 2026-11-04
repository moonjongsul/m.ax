#!/usr/bin/env python3
"""Add a top-down tray camera (`tray.mp4`) to a raw max_data_collect dataset.

The tray view is derived offline from `env.mp4`, with the same warp the live
node `max_ai_env_tray_span` applies to the env topic: the four tray corners
are perspective-flattened, rotated, and scaled to out_width x out_height.
Output size, rotation and (by default) the corners come from that node's
config, so a dataset built here matches what the policy sees at run time.

The source dataset is never modified. It is copied to `--out` (default
`<src>_tray`) and only the copy gains:

    episodes/*/tray.mp4                 N frames, same N as data.hdf5
    episodes/*/{start,end}_tray.jpg     when the episode has keyframes
    episodes/*/tasks.json               keyframes += the two jpgs
    episodes/*/data.hdf5 /meta          cameras += "tray"
    metadata.json                       cameras.tray (with the corners used)

There is no `stale` bit for tray: the bitmask is uint8 and every bit is
taken. Tray is a pure function of env, so env's bit covers it.

The env camera moved between collection days, so corners are per dataset.
Check them on a frame first, then pass them with --corners.

Usage:
    python3 add_tray_view.py /path/to/dataset                     # corners from config
    python3 add_tray_view.py /path/to/dataset --corners 405,72,601,101,617,322,343,281
    python3 add_tray_view.py /path/to/dataset --dry-run           # plan only
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import h5py
import numpy as np
import yaml

SRC_ROOT = Path(__file__).resolve().parents[2]  # ros2/src
for p in (SRC_ROOT / "max_ai", SRC_ROOT / "max_data_collect"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from max_ai.nodes.env_tray_span import warp_tray  # noqa: E402
from max_data_collect.writer.video_writer import VideoWriter  # noqa: E402

DEFAULT_TRAY_CONFIG = SRC_ROOT / "max_ai" / "config" / "env_tray_span_config.yaml"
CAMERA = "tray"
SOURCE_CAMERA = "env"


def load_tray_config(path: Path) -> dict:
    params = yaml.safe_load(path.read_text())["max_ai_env_tray_span"]["ros__parameters"]
    return {
        "corners": [float(v) for v in params["ref_corners"]],
        "proc_size": (int(params["proc_width"]), int(params["proc_height"])),
        "out_size": (int(params["out_width"]), int(params["out_height"])),
        "rotate": params["out_rotate"],
    }


def parse_corners(text: str) -> list[float]:
    values = [float(v) for v in text.split(",")]
    if len(values) != 8:
        raise argparse.ArgumentTypeError(f"need 8 numbers (TL TR BR BL as x,y), got {len(values)}")
    return values


def num_frames(episode: Path) -> int:
    with h5py.File(episode / "data.hdf5", "r") as f:
        return int(f["meta"].attrs["num_frames"])


def build_episode(episode: Path, cfg: dict, codec: str, cq: int, preset: str, hz: float) -> int:
    """Write tray.mp4 (+ keyframes) for one episode. Returns the frame count."""
    n = num_frames(episode)
    corners = np.float32(cfg["corners"]).reshape(4, 2)
    out_w, out_h = cfg["out_size"]

    cap = cv2.VideoCapture(str(episode / f"{SOURCE_CAMERA}.mp4"))
    if not cap.isOpened():
        raise RuntimeError(f"{episode.name}: cannot open {SOURCE_CAMERA}.mp4")
    writer = VideoWriter(None, episode / f"{CAMERA}.mp4", out_w, out_h, hz, codec, cq, preset)
    first = last = None
    scale = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if scale is None:
                # Corners are in proc-size pixels, as in the live node.
                h, w = frame.shape[:2]
                scale = np.float32([w / cfg["proc_size"][0], h / cfg["proc_size"][1]])
            tray = warp_tray(frame, corners * scale, out_w, out_h, rotate=cfg["rotate"])
            writer.write(tray)
            if first is None:
                first = tray
            last = tray
        written = writer.close()
    except BaseException:
        writer.abort()
        raise
    finally:
        cap.release()

    if written != n:
        raise RuntimeError(f"{episode.name}: {SOURCE_CAMERA}.mp4 has {written} frames, data.hdf5 has {n}")

    # Keyframes only where the recorder wrote them (older datasets have none).
    tasks_path = episode / "tasks.json"
    if (episode / f"start_{SOURCE_CAMERA}.jpg").exists():
        cv2.imwrite(str(episode / f"start_{CAMERA}.jpg"), first, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(episode / f"end_{CAMERA}.jpg"), last, [cv2.IMWRITE_JPEG_QUALITY, 90])
        tasks = json.loads(tasks_path.read_text())
        keyframes = tasks.get("keyframes", [])
        for name in (f"start_{CAMERA}.jpg", f"end_{CAMERA}.jpg"):
            if name not in keyframes:
                keyframes.append(name)
        tasks["keyframes"] = keyframes
        tasks_path.write_text(json.dumps(tasks, indent=2, ensure_ascii=False) + "\n")

    with h5py.File(episode / "data.hdf5", "r+") as f:
        cams = json.loads(f["meta"].attrs.get("cameras", "[]"))
        if CAMERA not in cams:
            cams.append(CAMERA)
            f["meta"].attrs["cameras"] = json.dumps(cams)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="raw dataset directory (read only)")
    ap.add_argument("--out", type=Path, help="output dataset directory (default: <src>_tray)")
    ap.add_argument("--corners", type=parse_corners,
                    help="x1,y1,...,x4,y4 (TL TR BR BL) in proc-size pixels; default: config ref_corners")
    ap.add_argument("--tray-config", type=Path, default=DEFAULT_TRAY_CONFIG)
    ap.add_argument("--codec", help="default: the dataset's video_codec")
    ap.add_argument("--cq", type=int, default=28, help="encoder quality, as the recorder's video_cq")
    ap.add_argument("--preset", default="p4")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = args.src.resolve()
    out = (args.out or src.with_name(src.name + "_tray")).resolve()
    if out == src:
        sys.exit("--out must differ from src; the source dataset is never modified")
    meta = json.loads((src / "metadata.json").read_text())
    if CAMERA in meta.get("cameras", {}):
        sys.exit(f"{src.name} already has a '{CAMERA}' camera")

    cfg = load_tray_config(args.tray_config)
    if args.corners:
        cfg["corners"] = args.corners
    codec = args.codec or meta.get("video_codec", "h264_nvenc")
    hz = float(meta["collect_hz"])
    episodes = sorted(p for p in (src / "episodes").iterdir() if (p / f"{SOURCE_CAMERA}.mp4").exists())

    print(f"{src.name} -> {out.name}: {len(episodes)} episodes, codec {codec} cq {args.cq}")
    print(f"  corners {cfg['corners']}  out {cfg['out_size'][0]}x{cfg['out_size'][1]} rotate {cfg['rotate']}")
    if args.dry_run:
        return 0
    if out.exists():
        sys.exit(f"{out} already exists; remove it first")

    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)

    def job(ep: Path) -> tuple[str, int]:
        return ep.name, build_episode(tmp / "episodes" / ep.name, cfg, codec, args.cq, args.preset, hz)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (name, n) in enumerate(pool.map(job, episodes), 1):
            print(f"  [{i}/{len(episodes)}] {name}: {n} frames", flush=True)

    meta["cameras"][CAMERA] = {
        "resolution": list(cfg["out_size"]),
        "rotate": 0,
        "derived_from": SOURCE_CAMERA,
        "corners": cfg["corners"],
        "corners_size": list(cfg["proc_size"]),
        "warp_rotate": cfg["rotate"],
    }
    (tmp / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    tmp.rename(out)
    print(f"done: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
