"""On-demand H.264 proxies for browser playback.

The recorder writes HEVC (`hevc_nvenc`), which no major browser decodes in a
`<video>` element on Linux. Rather than re-encode the dataset -- the originals
are the archival copy and must stay untouched -- each mp4 gets an H.264 proxy
generated on first request and cached on disk.

Frame count is preserved exactly (no resampling), so frame indices in the UI
stay valid against the HDF5 arrays.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import threading
from pathlib import Path

DEFAULT_PLAYABLE = ("h264", "avc1", "vp8", "vp9", "av1")
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def ffprobe_codec(path: Path) -> str | None:
    """Video codec name, or None if ffprobe is unavailable or the file is unreadable."""
    if not shutil.which("ffprobe"):
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=20,
        )
        return (out.stdout.strip() or None) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def needs_proxy(path: Path, video_cfg: dict | None = None) -> bool:
    cfg = video_cfg or {}
    mode = cfg.get("proxy", "auto")
    if mode == "never":
        return False
    if mode == "always":
        return True
    playable = {str(c).lower() for c in cfg.get("playable_codecs") or DEFAULT_PLAYABLE}
    codec = ffprobe_codec(path)
    # Unknown codec (no ffprobe): assume it is playable and let the browser try.
    return codec is not None and codec.lower() not in playable


def _packet_count(path: Path) -> int | None:
    """Number of video packets, or None if it cannot be determined.

    Counting packets (container-level, no decoding) is fast and, unlike the
    header duration, actually shrinks when a file is truncated.
    """
    if not shutil.which("ffprobe"):
        return None
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_packets", "-show_entries", "stream=nb_read_packets",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = res.stdout.strip().rstrip(",")
    return int(value) if res.returncode == 0 and value.isdigit() else None


def _is_complete(path: Path, expect_packets: int | None = None) -> bool:
    """True when `path` is a fully written, decodable proxy.

    Size alone is not enough, and neither is the header duration: `faststart`
    puts the moov atom up front, so a truncated file still reports the full
    duration while decoding only part of the frames. Comparing the packet
    count against the source is what actually catches a half-written encode
    (server restart, full disk) that the browser would reject mid-playback.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return False
    count = _packet_count(path)
    if count is None:          # no ffprobe: fall back to the size check
        return True
    if count == 0:
        return False
    if expect_packets:
        # Encoders may legitimately differ by a frame or two.
        return abs(count - expect_packets) <= 2
    return True


def _cache_path(src: Path, cache_root: Path) -> Path:
    st = src.stat()
    # Key on identity + mtime + size so a re-recorded episode invalidates itself.
    key = hashlib.sha1(f"{src}:{st.st_mtime_ns}:{st.st_size}".encode()).hexdigest()[:16]
    return cache_root / f"{src.stem}_{key}.mp4"


def _encoder_args(cfg: dict) -> list[str]:
    """Prefer the configured GPU encoder (~1s per episode), fall back to libx264."""
    encoder = str(cfg.get("encoder") or "h264_nvenc")
    try:
        available = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                                   capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        available = ""
    if encoder in available:
        if "nvenc" in encoder:
            return ["-c:v", encoder,
                    "-preset", str(cfg.get("nvenc_preset", "p4")),
                    "-cq", str(cfg.get("nvenc_cq", 26))]
        return ["-c:v", encoder]
    return ["-c:v", "libx264",
            "-preset", str(cfg.get("x264_preset", "veryfast")),
            "-crf", str(cfg.get("x264_crf", 23))]


def get_proxy(src: Path, cache_root: Path, video_cfg: dict | None = None) -> Path:
    """Return a browser-playable path for `src`, transcoding into the cache if needed.

    Falls back to the original on any ffmpeg failure -- a broken proxy should
    not make the episode unopenable.
    """
    cfg = video_cfg or {}
    if not needs_proxy(src, cfg):
        return src

    cache_root.mkdir(parents=True, exist_ok=True)
    dst = _cache_path(src, cache_root)
    expect = _packet_count(src)
    if _is_complete(dst, expect):
        return dst

    # One encode per file even if several <video> elements request it at once.
    with _locks_guard:
        lock = _locks.setdefault(str(dst), threading.Lock())
    with lock:
        if _is_complete(dst, expect):
            return dst
        # A leftover truncated proxy would otherwise be served forever.
        dst.unlink(missing_ok=True)
        # Unique per process so two servers on one dataset cannot collide.
        tmp = dst.with_suffix(f".part{os.getpid()}.mp4")
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src),
               *_encoder_args(cfg),
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(tmp)]
        # `finally` matters: a client that disconnects mid-encode unwinds this
        # thread, and without it the .part file is orphaned in the cache.
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            if res.returncode != 0 or not _is_complete(tmp, expect):
                _log_failure(src, (res.stderr or "").strip()[:300] or "incomplete output")
                return src
            tmp.replace(dst)
            return dst
        except (OSError, subprocess.SubprocessError) as exc:
            _log_failure(src, f"{type(exc).__name__}: {exc}")
            return src
        finally:
            tmp.unlink(missing_ok=True)


def sweep_stale(cache_root: Path) -> int:
    """Delete leftover .part files from a previous run. Returns how many."""
    if not cache_root.is_dir():
        return 0
    removed = 0
    for leftover in cache_root.glob("*.part*.mp4"):
        try:
            leftover.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def _log_failure(src: Path, detail: str) -> None:
    """Transcoding failed, so the browser is about to get unplayable HEVC.
    Say so in the log rather than leaving a silent black video element."""
    print(f"[max_data_editor] transcode FAILED for {src.name}: {detail}\n"
          f"[max_data_editor] serving the original; the browser will likely "
          f"report a decode error", flush=True)
