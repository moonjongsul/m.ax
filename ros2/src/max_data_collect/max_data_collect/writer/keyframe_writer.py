"""Episode keyframes: the first and last kept frame, as JPEGs.

Written next to the mp4s:

    start_<camera>.jpg   end_<camera>.jpg     one per camera
    start_composite.jpg  end_composite.jpg    the same frames side by side

The frames come from the writer's own queue, so they are the exact
arrays that became video frame 0 and frame N-1 -- already decoded,
rotated and resized. Nothing is re-decoded and the sampling loop is not
touched.

"First" means the first frame after warmup, because warmup frames are
never enqueued; "last" means the last frame before STOP. Both are held
as references only: one frame per camera per end, not a copy of the
episode.
"""

import cv2
import numpy as np


class KeyframeWriter:
    """Holds the first/last frame per camera and writes them on finish."""

    def __init__(self, logger, episode_dir, camera_names: list[str],
                 composite_order: list[str], jpeg_quality: int):
        self._log = logger
        self._dir = episode_dir
        self._cameras = list(camera_names)
        self._order = [n for n in composite_order if n in self._cameras]
        self._params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        self._start: dict[str, np.ndarray] = {}
        self._last: dict[str, np.ndarray] = {}

    def observe(self, images: dict) -> None:
        """Called once per row, on the writer thread.

        Cameras are tracked independently: a camera whose very first
        frame failed to decode still gets a start image from the first
        frame that did, rather than being dropped for the episode.
        """
        for name in self._cameras:
            frame = images.get(name)
            if frame is None:
                continue
            if name not in self._start:
                self._start[name] = frame
            self._last[name] = frame

    def write(self) -> list[str]:
        """Write every keyframe file. Returns the filenames written."""
        written = []
        for tag, frames in (("start", self._start), ("end", self._last)):
            for name in self._cameras:
                frame = frames.get(name)
                if frame is None:
                    self._log.warning(
                        f"[collect] no {tag} keyframe for camera '{name}'; "
                        "it never delivered a decodable frame"
                    )
                    continue
                written.append(self._save(f"{tag}_{name}.jpg", frame))

            composite = self._compose(frames)
            if composite is not None:
                written.append(self._save(f"{tag}_composite.jpg", composite))
        return [w for w in written if w]

    # ── internals ──────────────────────────────────────────────────────

    def _compose(self, frames: dict):
        """Lay the frames out left to right in `composite_order`.

        Cameras may differ in resolution, so each panel is scaled to a
        common height -- the tallest -- preserving its aspect ratio. A
        camera with no frame is skipped rather than leaving a gap, so the
        composite stays a picture of what was actually recorded.
        """
        panels = [frames[n] for n in self._order if frames.get(n) is not None]
        if not panels:
            return None
        height = max(p.shape[0] for p in panels)
        scaled = []
        for panel in panels:
            if panel.shape[0] != height:
                width = max(1, round(panel.shape[1] * height / panel.shape[0]))
                panel = cv2.resize(panel, (width, height),
                                   interpolation=cv2.INTER_AREA)
            scaled.append(panel)
        return np.hstack(scaled)

    def _save(self, filename: str, image) -> str:
        path = self._dir / filename
        try:
            if not cv2.imwrite(str(path), image, self._params):
                raise RuntimeError("cv2.imwrite returned false")
        except Exception as exc:                    # noqa: BLE001
            # A missing thumbnail must never cost a recorded episode.
            self._log.warning(f"[collect] cannot write {filename}: {exc}")
            return ""
        return filename
