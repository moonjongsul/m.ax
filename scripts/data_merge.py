"""기존 joint-space LeRobot 데이터셋(FROM)의 에피소드를 신규 통합 데이터셋(TO)으로
이전/병합한다.

- FROM 의 모든 에피소드는 'flip_object' task 로 간주하고, yaml 에 정의된 해당 task 의
  paraphrase pool 에서 frame 마다 랜덤 샘플링하여 재배정한다.
- TO 가 존재하면 append, 없으면 신규 생성.
- repo_id / episode_index / task_index 등 식별자는 모두 TO 기준으로 재작성된다.
- 비디오는 LeRobot 이 re-encoding (save_episode 경로).
- Hub push 없음: 로컬에만 저장.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "thirdparty" / "lerobot" / "src"))
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402


FROM_DB = "/workspace/m.ax/datasets/flip_object"
TO_DB = "/workspace/m.ax/datasets/manufacturing_kitting_dataset"
DB_YAML = "/workspace/m.ax/config/data_collect_config.yaml"
MIGRATE_TASK_ID = "flip_object"


def _load_pool(yaml_path: str, task_id: str) -> list[str]:
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    task_prompt = cfg["data_collect"]["task_prompt"]
    if task_id not in task_prompt:
        raise KeyError(f"task_id '{task_id}' not in {yaml_path}")
    pool = [str(p).strip() for p in (task_prompt[task_id] or []) if str(p).strip()]
    if not pool:
        pool = [task_id.replace("_", " ").replace("-", " ")]
    return pool


def _tensor_to_numpy(x):
    """torch.Tensor → numpy; 그 외는 그대로."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
    except ImportError:
        pass
    return x


def _load_target_dataset(src: LeRobotDataset, to_root: Path, to_repo_id: str) -> LeRobotDataset:
    """TO 가 이미 있으면 로드, 없으면 FROM 의 features/fps 로 생성."""
    info_file = to_root / "meta" / "info.json"
    tasks_file = to_root / "meta" / "tasks.parquet"
    is_complete = info_file.exists() and tasks_file.exists()

    if is_complete:
        print(f"[to] 기존 데이터셋 로드 (append): {to_root}")
        dst = LeRobotDataset(repo_id=to_repo_id, root=to_root)
        dst.start_image_writer(num_processes=0, num_threads=4)
        return dst

    print(f"[to] 신규 생성: {to_root}")
    return LeRobotDataset.create(
        repo_id=to_repo_id,
        fps=src.fps,
        features=src.features,
        root=to_root,
        robot_type=src.meta.robot_type,
        use_videos=True,
        image_writer_threads=4,
    )


def migrate_episode(
    src: LeRobotDataset,
    dst: LeRobotDataset,
    ep_idx: int,
    pool: list[str],
    rng: random.Random,
) -> int:
    """FROM 의 한 에피소드를 읽어 TO 에 add_frame + save_episode."""
    ep_meta = src.meta.episodes[ep_idx]
    start = int(ep_meta["dataset_from_index"])
    end = int(ep_meta["dataset_to_index"])

    for abs_idx in range(start, end):
        item = src[abs_idx]
        frame: dict = {}
        for key, ft in src.features.items():
            if key in ("index", "episode_index", "task_index", "frame_index", "timestamp"):
                continue
            value = item[key]
            if ft["dtype"] in ("image", "video"):
                arr = _tensor_to_numpy(value)
                # src.__getitem__ 은 video frame 을 float32 [0,1] CHW 로 반환.
                # image_writer 는 HWC uint8 PNG 저장을 원함.
                if arr.dtype != np.uint8:
                    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
                if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] != arr.shape[-1]:
                    # CHW → HWC
                    arr = np.transpose(arr, (1, 2, 0))
                frame[key] = arr
            else:
                frame[key] = _tensor_to_numpy(value)
        frame["task"] = rng.choice(pool)
        dst.add_frame(frame)

    dst.save_episode()
    return end - start


def main() -> None:
    from_root = Path(FROM_DB)
    to_root = Path(TO_DB)
    to_repo_id = to_root.name

    if not (from_root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"FROM 에 info.json 이 없음: {from_root}")

    pool = _load_pool(DB_YAML, MIGRATE_TASK_ID)
    print(f"[pool] task='{MIGRATE_TASK_ID}' size={len(pool)}")

    # FROM 로드: repo_id 는 TO 와 달라도 상관없음 (로컬 root 기준 로드).
    src = LeRobotDataset(repo_id=to_repo_id, root=from_root)
    print(f"[from] episodes={src.num_episodes} frames={src.num_frames}")

    to_root.parent.mkdir(parents=True, exist_ok=True)
    dst = _load_target_dataset(src, to_root, to_repo_id)
    print(f"[to] before: episodes={dst.num_episodes} frames={dst.num_frames}")

    rng = random.Random()
    total_frames = 0
    for ep_idx in range(src.num_episodes):
        n = migrate_episode(src, dst, ep_idx, pool, rng)
        total_frames += n
        print(f"  ep {ep_idx + 1}/{src.num_episodes} migrated ({n} frames)")

    dst.finalize()
    print(f"[to] after : episodes={dst.num_episodes} frames={dst.num_frames}")
    print(f"[done] migrated {src.num_episodes} episodes, {total_frames} frames → {to_root}")


if __name__ == "__main__":
    main()
