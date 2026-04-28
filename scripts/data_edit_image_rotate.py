"""LeRobotDataset 후처리 편집 스크립트.

현재 동작: 지정한 카메라(observation.images.wrist_rear)의 모든 비디오 파일을
180도 회전하여 in-place 덮어쓴다.

- 인코딩: lerobot 기본과 동일 (libsvtav1, crf=30, preset=12, yuv420p)
- 백업: <root>/videos.bak/<camera>/ 에 원본 보존 (재실행 시 그대로 덮어쓰지 않음)
- 멱등성 없음: 두 번 실행하면 원래 방향으로 돌아간다. 백업 폴더 존재 시 거부.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# ── 하드코딩 설정 ──────────────────────────────────────────────────────
REPO_ID = "manufacturing_kitting_dataset"
DATASET_ROOT = Path("/workspace/m.ax/datasets") / REPO_ID
TARGET_CAMERA = "observation.images.wrist_rear"

# lerobot 인코딩 디폴트와 일치 (thirdparty/lerobot/src/lerobot/datasets/video_utils.py)
VCODEC = "libsvtav1"
CRF = 30
PRESET = 12
PIX_FMT = "yuv420p"


def rotate_video_180(src: Path, dst: Path) -> None:
    """src 비디오를 180도 회전해서 dst로 인코딩."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-vf", "transpose=2,transpose=2",  # 180° (CCW90 두 번)
        "-c:v", VCODEC,
        "-pix_fmt", PIX_FMT,
        "-crf", str(CRF),
        "-preset", str(PRESET),
        "-an",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    cam_dir = DATASET_ROOT / "videos" / TARGET_CAMERA
    if not cam_dir.exists():
        raise SystemExit(f"카메라 폴더 없음: {cam_dir}")

    backup_dir = DATASET_ROOT / "videos.bak" / TARGET_CAMERA
    if backup_dir.exists():
        raise SystemExit(
            f"백업 폴더가 이미 존재: {backup_dir}\n"
            f"이전 실행 흔적이거나 이미 회전된 상태일 수 있다.\n"
            f"의도한 재실행이라면 백업 폴더를 먼저 삭제/이동하라."
        )

    video_files = sorted(cam_dir.rglob("*.mp4"))
    if not video_files:
        raise SystemExit(f"비디오 파일 없음: {cam_dir}")

    print(f"[edit] target camera: {TARGET_CAMERA}")
    print(f"[edit] {len(video_files)} files to rotate 180°")
    print(f"[edit] backup → {backup_dir}")

    # 1) 원본 백업 (디렉토리 구조 보존)
    backup_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(cam_dir, backup_dir)
    print("[edit] backup 완료")

    # 2) 각 파일 회전 → 같은 위치에 덮어쓰기
    for i, src in enumerate(video_files, 1):
        rel = src.relative_to(cam_dir)
        tmp = src.with_suffix(".rot.mp4")
        print(f"[{i}/{len(video_files)}] {rel}  ", end="", flush=True)
        try:
            rotate_video_180(src, tmp)
        except subprocess.CalledProcessError as e:
            if tmp.exists():
                tmp.unlink()
            print(f"\n[edit] ffmpeg 실패: {rel}")
            return e.returncode
        tmp.replace(src)
        size_mb = src.stat().st_size / (1024 * 1024)
        print(f"ok ({size_mb:.1f} MB)")

    print(f"\n[edit] 완료. 원본 백업: {backup_dir}")
    print(f"[edit] 결과 확인 후 백업을 지우려면: rm -rf {backup_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
