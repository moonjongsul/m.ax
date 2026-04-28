"""특정 에피소드 인덱스부터의 task 프롬프트를 다른 풀로 교체.

상황: data_collect.py에서 active task 전환을 깜박해서 ep 119+ 가 실제로는
kit_object 작업인데 flip_object 풀의 paraphrase로 저장됐다.

이 스크립트는 ep `START_EP` 이상에 대해, 각 프레임의 task_index를 config의
`kit_object` 풀 안에서 *프레임 단위 무작위*로 재배정한다 (data_collect.py와
동일하게 paraphrase가 프레임마다 섞이는 분포를 유지).

수정 대상:
- meta/tasks.parquet         : 새 task 가 있으면 추가 (기존 인덱스는 유지)
- data/**/*.parquet          : 해당 ep 들의 task_index 컬럼 재할당
- meta/episodes/**/*.parquet : 해당 ep 행의 tasks 컬럼(=unique paraphrase list) 갱신
- meta/info.json             : total_tasks 갱신

멱등성 없음. 실행 전 데이터셋 백업 권장.
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "thirdparty" / "lerobot" / "src"))

import pandas as pd
import yaml

# ── 하드코딩 설정 ──────────────────────────────────────────────────────
REPO_ID = "manufacturing_kitting_dataset"
DATASET_ROOT = Path("/workspace/m.ax/datasets") / REPO_ID
CONFIG_PATH = Path("/workspace/m.ax/config/proj_gt_kitting_config.yaml")
START_EP = 106           # 이 인덱스부터 끝까지가 교체 대상
TARGET_TASK_ID = "kit_object"
RANDOM_SEED = 0          # 재현성


def main() -> int:
    # ── 1) config 에서 새 풀 로드 ───────────────────────────────────
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    pools = cfg["data_collect"]["task_prompt"]
    if TARGET_TASK_ID not in pools:
        raise SystemExit(f"task_id {TARGET_TASK_ID!r} 가 config 에 없음. keys={list(pools)}")
    new_pool = [str(p).strip() for p in pools[TARGET_TASK_ID] if str(p).strip()]
    if not new_pool:
        raise SystemExit(f"{TARGET_TASK_ID} 풀이 비어있음")
    print(f"[edit] target pool ({TARGET_TASK_ID}): {len(new_pool)} paraphrases")
    print(f"{new_pool}")

    # """


    # ── 2) 기존 tasks.parquet 로드 → 새 paraphrase 추가 ──────────────
    tasks_path = DATASET_ROOT / "meta" / "tasks.parquet"
    tasks_df = pd.read_parquet(tasks_path)
    # tasks_df: index=task name, column 'task_index'
    existing_tasks: dict[str, int] = {name: int(row["task_index"]) for name, row in tasks_df.iterrows()}
    next_idx = (max(existing_tasks.values()) + 1) if existing_tasks else 0

    added: list[tuple[str, int]] = []
    for p in new_pool:
        if p not in existing_tasks:
            existing_tasks[p] = next_idx
            added.append((p, next_idx))
            next_idx += 1
    if added:
        print(f"[edit] tasks.parquet 에 {len(added)} 개 paraphrase 신규 등록")
        for p, i in added:
            print(f"  + [{i}] {p}")
    else:
        print("[edit] 모든 paraphrase 가 이미 tasks.parquet 에 존재")

    new_pool_indices = [existing_tasks[p] for p in new_pool]
    idx_to_name = {v: k for k, v in existing_tasks.items()}

    # ── 3) 데이터 parquet 의 task_index 재할당 ─────────────────────
    rng = random.Random(RANDOM_SEED)
    data_dir = DATASET_ROOT / "data"
    affected_data_files: list[Path] = []
    total_frames_changed = 0

    for parquet_path in sorted(data_dir.rglob("*.parquet")):
        df = pd.read_parquet(parquet_path)
        mask = df["episode_index"] >= START_EP
        if not mask.any():
            continue
        n = int(mask.sum())
        new_indices = [rng.choice(new_pool_indices) for _ in range(n)]
        df.loc[mask, "task_index"] = new_indices
        df.to_parquet(parquet_path, index=False)
        affected_data_files.append(parquet_path)
        total_frames_changed += n
        print(f"  data: {parquet_path.relative_to(DATASET_ROOT)}  +{n} frames updated")

    if total_frames_changed == 0:
        print(f"[edit] ep>={START_EP} 인 데이터가 없음. 종료.")
        return 0
    print(f"[edit] data parquet: {total_frames_changed} frames 재할당")

    # ── 4) episodes 메타의 tasks 컬럼 갱신 ──────────────────────────
    # tasks 컬럼 = 그 에피소드에 등장하는 unique paraphrase 리스트.
    # 위에서 새로 쓴 task_index 분포로부터 재계산.
    ep_to_paraphrases: dict[int, list[str]] = {}
    for parquet_path in affected_data_files:
        df = pd.read_parquet(parquet_path)
        sub = df[df["episode_index"] >= START_EP]
        for ep, g in sub.groupby("episode_index"):
            uniq = sorted({idx_to_name[int(i)] for i in g["task_index"].unique()})
            ep_to_paraphrases[int(ep)] = uniq

    episodes_dir = DATASET_ROOT / "meta" / "episodes"
    for parquet_path in sorted(episodes_dir.rglob("*.parquet")):
        ep_df = pd.read_parquet(parquet_path)
        mask = ep_df["episode_index"] >= START_EP
        if not mask.any():
            continue
        new_col = ep_df["tasks"].tolist()
        for i, ep in enumerate(ep_df["episode_index"].tolist()):
            ep = int(ep)
            if ep >= START_EP and ep in ep_to_paraphrases:
                new_col[i] = ep_to_paraphrases[ep]
        ep_df["tasks"] = new_col
        ep_df.to_parquet(parquet_path, index=False)
        affected_eps = ep_df.loc[mask, "episode_index"]
        print(f"  meta: {parquet_path.relative_to(DATASET_ROOT)}  "
              f"ep{int(affected_eps.min())}-{int(affected_eps.max())} tasks 갱신")

    # ── 5) tasks.parquet 갱신 ───────────────────────────────────────
    if added:
        new_tasks_df = pd.DataFrame(
            {"task_index": list(existing_tasks.values())},
            index=pd.Index(list(existing_tasks.keys()), name="task"),
        )
        new_tasks_df = new_tasks_df.sort_values("task_index")
        shutil.copy2(tasks_path, tasks_path.with_suffix(".bak.parquet"))
        new_tasks_df.to_parquet(tasks_path, index=True)
        print(f"[edit] tasks.parquet 갱신 (총 {len(new_tasks_df)} tasks)")

    # ── 6) info.json 갱신 ───────────────────────────────────────────
    info_path = DATASET_ROOT / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["total_tasks"] = len(existing_tasks)
    info_path.write_text(json.dumps(info, indent=4))
    print(f"[edit] info.json: total_tasks={info['total_tasks']}")

    # ── 7) 검증 ─────────────────────────────────────────────────────
    print("[edit] 검증: LeRobotDataset 로딩")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id=REPO_ID, root=DATASET_ROOT)
    print(f"  num_episodes={ds.num_episodes}, num_frames={ds.num_frames}")

    full = pd.concat(
        [pd.read_parquet(p) for p in sorted((DATASET_ROOT / "data").rglob("*.parquet"))],
        ignore_index=True,
    )
    sample_eps = [START_EP, START_EP + 1, ds.num_episodes - 1]
    for ep in sample_eps:
        if ep >= ds.num_episodes:
            continue
        idxs = sorted(full[full["episode_index"] == ep]["task_index"].unique().tolist())
        names = [idx_to_name[int(i)] for i in idxs]
        ok = all(n in new_pool for n in names)
        print(f"  ep{ep:3d}: indices={idxs}  ok={ok}")
        if not ok:
            print(f"    !! 풀 외 paraphrase 발견: {[n for n in names if n not in new_pool]}")
    print("[edit] 완료")
    return 0
    # """


if __name__ == "__main__":
    sys.exit(main())
