"""data_collect 중단으로 손상된 LeRobotDataset을 ep 0-118로 되돌리는 일회성 복구 스크립트.

상황:
- meta/episodes/chunk-000/file-001.parquet (ep 119-129) footer 미작성 → 손상
- data/chunk-000/file-002.parquet (ep 119-129) footer 미작성 → 손상
- info.json 카운터(total_episodes=130, total_frames=51066, splits.train=0:130) 잘못 표기
- 카메라별 마지막 비디오 파일은 ep 119-129 전용이라 통째로 삭제 가능

이 스크립트는 손상 파일 + ep 119-129 비디오 파일을 제거하고 info.json을 재계산,
stats.json을 episodes 메타에서 다시 aggregate한다. 실행 후 데이터셋 = ep 0-118.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "thirdparty" / "lerobot" / "src"))

import numpy as np
import pandas as pd
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.io_utils import write_stats

DATASET_ROOT = Path("/workspace/m.ax/datasets/manufacturing_kitting_dataset")
KEEP_LAST_EP = 118  # inclusive
EXPECTED_TOTAL_FRAMES = 48023

CAMERAS_LAST_FILE_KEEP = {
    "observation.images.wrist_front": 3,
    "observation.images.wrist_rear":  2,
    "observation.images.front_view":  4,
    "observation.images.side_view":   6,
}


def remove_path(p: Path) -> None:
    if p.exists():
        print(f"  rm {p.relative_to(DATASET_ROOT)}")
        p.unlink()
    else:
        print(f"  (skip, not found) {p.relative_to(DATASET_ROOT)}")


def main() -> None:
    if not DATASET_ROOT.exists():
        raise SystemExit(f"Dataset not found: {DATASET_ROOT}")

    # ── 1) 손상된 ep119-129 parquet 삭제 ────────────────────────────
    print("[1] 손상된 parquet 삭제")
    remove_path(DATASET_ROOT / "data/chunk-000/file-002.parquet")
    remove_path(DATASET_ROOT / "meta/episodes/chunk-000/file-001.parquet")

    # ── 2) ep119-129 전용 비디오 파일 삭제 ──────────────────────────
    print("[2] ep119-129 전용 비디오 파일 삭제")
    for cam_key, last_keep in CAMERAS_LAST_FILE_KEEP.items():
        cam_dir = DATASET_ROOT / "videos" / cam_key / "chunk-000"
        if not cam_dir.exists():
            print(f"  (skip) {cam_dir} 없음")
            continue
        for f in sorted(cam_dir.glob("file-*.mp4")):
            file_idx = int(f.stem.split("-")[1])
            if file_idx > last_keep:
                remove_path(f)

    # ── 3) episodes 메타 재로드 + 정합성 검증 ───────────────────────
    print("[3] episodes 메타 검증")
    ep_meta = pd.read_parquet(DATASET_ROOT / "meta/episodes/chunk-000/file-000.parquet")
    if int(ep_meta["episode_index"].max()) != KEEP_LAST_EP:
        raise SystemExit(f"unexpected episodes parquet: max ep={ep_meta['episode_index'].max()}")
    actual_frames = int(ep_meta["length"].sum())
    if actual_frames != EXPECTED_TOTAL_FRAMES:
        raise SystemExit(f"frame count mismatch: {actual_frames} != {EXPECTED_TOTAL_FRAMES}")
    print(f"  ep range: 0-{KEEP_LAST_EP} ({len(ep_meta)} eps), total_frames: {actual_frames}")

    # ── 4) info.json 갱신 ───────────────────────────────────────────
    print("[4] info.json 갱신")
    info_path = DATASET_ROOT / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["total_episodes"] = len(ep_meta)
    info["total_frames"] = actual_frames
    info["splits"] = {"train": f"0:{len(ep_meta)}"}
    info_path.write_text(json.dumps(info, indent=4))
    print(f"  total_episodes={info['total_episodes']}, total_frames={info['total_frames']}, splits={info['splits']}")

    # ── 5) stats.json 재계산 (episodes parquet의 per-ep stats를 aggregate) ──
    print("[5] stats.json 재계산")
    stats_columns = [c for c in ep_meta.columns if c.startswith("stats/")]
    feature_keys: set[str] = set()
    for c in stats_columns:
        # "stats/<feature>/<stat>"
        parts = c.split("/", 2)
        if len(parts) == 3:
            feature_keys.add(parts[1])
        elif len(parts) == 2:
            # 예외 케이스 없을 것이지만 방어
            pass

    # 정확히는 마지막 한 글자가 stat이고 앞이 feature. video 키처럼 path에 '/'가 들어가는 feature가 있어
    # 단순 split이 부족하다. 아래에서 다시 안전하게 파싱.
    feature_keys = set()
    stat_suffixes = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
    for c in stats_columns:
        rest = c[len("stats/"):]
        for suf in stat_suffixes:
            if rest.endswith("/" + suf):
                feature_keys.add(rest[: -(len(suf) + 1)])
                break

    print(f"  features: {sorted(feature_keys)}")

    per_ep_stats: list[dict[str, dict[str, np.ndarray]]] = []
    for _, row in ep_meta.iterrows():
        ep_stats: dict[str, dict[str, np.ndarray]] = {}
        for feat in feature_keys:
            is_image = "image" in feat
            d: dict[str, np.ndarray] = {}
            for suf in stat_suffixes:
                col = f"stats/{feat}/{suf}"
                if col in ep_meta.columns:
                    val = row[col]
                    if isinstance(val, np.ndarray):
                        arr = val
                    elif isinstance(val, (list, tuple)):
                        arr = np.asarray(val)
                    else:
                        arr = np.asarray([val])
                    arr = np.asarray(arr, dtype=np.float32 if suf != "count" else np.int64)

                    if suf == "count":
                        arr = arr.reshape(1)
                    elif is_image:
                        # 이미지 feature는 (C,1,1) 형태 강제
                        arr = arr.reshape(-1, 1, 1)
                    elif arr.ndim == 0:
                        arr = arr.reshape(1)

                    d[suf] = arr
            ep_stats[feat] = d
        per_ep_stats.append(ep_stats)

    aggregated = aggregate_stats(per_ep_stats)
    write_stats(aggregated, DATASET_ROOT)
    print("  stats.json 재작성 완료")

    # ── 6) 최종 검증 — LeRobotDataset 로딩 가능한지 확인 ────────────
    print("[6] LeRobotDataset 로딩 검증")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id="manufacturing_kitting_dataset", root=DATASET_ROOT)
    print(f"  loaded: num_episodes={ds.num_episodes}, num_frames={ds.num_frames}")
    if ds.num_episodes != KEEP_LAST_EP + 1:
        raise SystemExit("validation failed")
    print("OK")


if __name__ == "__main__":
    main()
