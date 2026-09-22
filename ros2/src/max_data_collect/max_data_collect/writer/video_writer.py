"""One ffmpeg subprocess per camera, fed raw BGR frames over stdin.

Every mp4 ends up with exactly N frames -- the same N as the HDF5 arrays
-- so frame i of any video is row i of the dataset. That invariant is why
no frame-index column is stored. A tick where a camera published nothing
repeats the previous frame (inter-frame compression makes a repeat nearly
free); a camera that has published nothing at all yet gets black.
"""

import subprocess

import numpy as np


class VideoWriter:
    """Encodes one camera stream to mp4."""

    def __init__(self, logger, path, width: int, height: int, fps: float,
                 codec: str, cq: int, preset: str):
        self._log = logger
        self._path = path
        self._size = (width, height)
        self._last: np.ndarray | None = None
        self._written = 0

        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{fps:g}",
            "-i", "-",
            "-c:v", codec, "-pix_fmt", "yuv420p",
        ]
        # Quality flag differs between the NVENC and software encoders.
        cmd += (["-cq", str(cq), "-preset", preset] if codec.endswith("_nvenc")
                else ["-crf", str(cq)])
        # H.264 is written to be played in a browser (the editor and the
        # LeRobot visualizer both use a <video> element), so ask for the
        # profile players expect and move the moov atom to the front -- a
        # trailing moov makes a player download the whole file before the
        # first frame appears.
        if codec in ("h264_nvenc", "libx264"):
            cmd += ["-profile:v", "high", "-movflags", "+faststart"]
        cmd.append(str(path))

        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame) -> None:
        """Append one frame, substituting the previous one if absent."""
        if frame is None:
            frame = self._last
        if frame is None:
            frame = np.zeros((self._size[1], self._size[0], 3), np.uint8)
        elif (frame.shape[1], frame.shape[0]) != self._size:
            raise ValueError(
                f"frame is {frame.shape[1]}x{frame.shape[0]}, expected "
                f"{self._size[0]}x{self._size[1]}"
            )
        self._last = frame
        self._proc.stdin.write(frame.tobytes())
        self._written += 1

    def close(self) -> int:
        """Flush and wait for the encoder. Returns the frame count."""
        if self._proc is None:
            return self._written
        try:
            self._proc.stdin.close()
        except BrokenPipeError:
            pass
        code = self._proc.wait()
        if code != 0:
            err = self._proc.stderr.read().decode(errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed for {self._path}: {err[:400]}")
        self._proc = None
        return self._written

    def abort(self) -> None:
        if self._proc is None:
            return
        self._proc.kill()
        self._proc.wait()
        self._proc = None

    @property
    def written(self) -> int:
        return self._written


def probe_codec(codec: str) -> bool:
    """True if ffmpeg can actually encode with `codec` on this machine."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, timeout=15,
        ).stdout.decode(errors="replace")
    except (OSError, subprocess.SubprocessError):
        return False
    return any(line.split()[1:2] == [codec] for line in out.splitlines()
               if line.startswith(" ") and len(line.split()) > 1)
