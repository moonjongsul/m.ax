"""tasks.parquet 을 26개 → 4개로 축소.

기존 prompt 풀이 동작 단위로 너무 잘게 쪼개져 있어 학습/평가에서 다루기 번거로움.
모든 prompt 를 다음 4개 중 하나로 매핑한다:

  0  flip part
  1  flip object
  2  kit object
  3  kit part

매핑 정책 (사용자 지정):
  - 기존 idx 0~10  → flip 그룹 (part / object 키워드로 0 또는 1)
  - 기존 idx 11~25 → kit  그룹 (part / object 키워드로 3 또는 2)

수정 대상 (기존 data_edit_prompt.py 와 동일 패턴):
  - meta/tasks.parquet         : 4행으로 재작성
  - data/**/*.parquet          : task_index 컬럼 전체 리매핑
  - meta/episodes/**/*.parquet : tasks 컬럼(에피소드별 paraphrase 리스트) 재계산
  - meta/info.json             : total_tasks=4

멱등성 없음 (한 번 실행하면 옛 26-task 인덱스는 사라짐). 백업본 .bak 로 저장.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "thirdparty" / "lerobot" / "src"))

import pandas as pd

REPO_ID = "manufacturing_kitting_dataset"
DATASET_ROOT = Path("/workspace/m.ax/datasets") / REPO_ID

NEW_TASKS = ["flip part", "flip object", "kit object", "kit part"]
NEW_NAME_TO_IDX = {name: i for i, name in enumerate(NEW_TASKS)}

# 26개 기존 prompt 문자열 → 새 인덱스 (0..3)
# part/object 키워드로 분류. "component" 는 object 그룹.
OLD_NAME_TO_NEW_IDX: dict[str, int] = {
    # 0..10  flip 그룹
    "flip the object":            1,  # object → flip object
    "turn object upside down":    1,
    "flip the part":              0,  # part   → flip part
    "flip the part over":         0,
    "flip the component":         1,  # component=object 처리
    "flip object":                1,
    "flip object over":           1,
    "flip part":                  0,
    "invert the part":            0,
    "turn the part over":         0,
    "turn the part upside down":  0,
    # 11..25 kit 그룹
    "fit part in tray":               3,  # part   → kit part
    "kit object":                     2,  # object → kit object
    "insert object into kit":         2,
    "position object in tray":        2,
    "kit object into tray":           2,
    "kit part":                       3,
    "align and place object in tray": 2,
    "set object in tray":             2,
    "kit object in tray":             2,
    "place object in tray":           2,
    "kit part into tray":             3,
    "nest object in tray":            2,
    "fit object into kit slot":       2,
    "seat object in tray":            2,
    "place object into kit":          2,
}


def main() -> int:
    # ── 1) 현재 tasks.parquet 로드 후 매핑 dict (old_idx → new_idx) 구축 ──
    tasks_path = DATASET_ROOT / "meta" / "tasks.parquet"
    old_tasks_df = pd.read_parquet(tasks_path)
    # index = task name, column 'task_index' = old index
    old_idx_to_new_idx: dict[int, int] = {}
    unmapped: list[str] = []
    for name, row in old_tasks_df.iterrows():
        old_idx = int(row["task_index"])
        if name not in OLD_NAME_TO_NEW_IDX:
            unmapped.append(name)
            continue
        old_idx_to_new_idx[old_idx] = OLD_NAME_TO_NEW_IDX[name]
    if unmapped:
        raise SystemExit(f"매핑되지 않은 prompt: {unmapped}")
    print(f"[simplify] 매핑 구축: {len(old_idx_to_new_idx)} old indices → {len(NEW_TASKS)} new indices")

    # ── 2) data/**/*.parquet 의 task_index 리매핑 ──────────────────
    data_dir = DATASET_ROOT / "data"
    total_frames = 0
    for parquet_path in sorted(data_dir.rglob("*.parquet")):
        df = pd.read_parquet(parquet_path)
        before_unique = sorted(df["task_index"].unique().tolist())
        df["task_index"] = df["task_index"].map(old_idx_to_new_idx).astype("int64")
        if df["task_index"].isna().any():
            raise SystemExit(f"{parquet_path}: 매핑 실패 (NaN). before_unique={before_unique}")
        df.to_parquet(parquet_path, index=False)
        total_frames += len(df)
        print(f"  data: {parquet_path.relative_to(DATASET_ROOT)}  "
              f"frames={len(df)}  old_uniq={before_unique} → new_uniq={sorted(df['task_index'].unique().tolist())}")
    print(f"[simplify] data parquet 리매핑 완료: 총 {total_frames} frames")

    # ── 3) episodes 메타의 tasks 컬럼 재계산 ──────────────────────
    # tasks 컬럼 = 해당 에피소드에 등장하는 unique paraphrase 리스트.
    # 새 task_index 분포로부터 다시 계산.
    ep_to_new_paraphrases: dict[int, list[str]] = {}
    for parquet_path in sorted(data_dir.rglob("*.parquet")):
        df = pd.read_parquet(parquet_path, columns=["episode_index", "task_index"])
        for ep, g in df.groupby("episode_index"):
            uniq_new_idx = sorted(int(i) for i in g["task_index"].unique())
            ep_to_new_paraphrases[int(ep)] = [NEW_TASKS[i] for i in uniq_new_idx]

    episodes_dir = DATASET_ROOT / "meta" / "episodes"
    for parquet_path in sorted(episodes_dir.rglob("*.parquet")):
        ep_df = pd.read_parquet(parquet_path)
        new_col = [ep_to_new_paraphrases[int(ep)] for ep in ep_df["episode_index"].tolist()]
        ep_df["tasks"] = new_col
        ep_df.to_parquet(parquet_path, index=False)
        eps = ep_df["episode_index"]
        print(f"  meta: {parquet_path.relative_to(DATASET_ROOT)}  "
              f"ep{int(eps.min())}-{int(eps.max())} tasks 갱신")

    # ── 4) tasks.parquet 재작성 (백업 후) ────────────────────────
    shutil.copy2(tasks_path, tasks_path.with_suffix(".bak.parquet"))
    new_tasks_df = pd.DataFrame(
        {"task_index": list(range(len(NEW_TASKS)))},
        index=pd.Index(NEW_TASKS, name="task"),
    )
    new_tasks_df.to_parquet(tasks_path, index=True)
    print(f"[simplify] tasks.parquet 재작성 (총 {len(new_tasks_df)} tasks):")
    print(new_tasks_df.to_string())

    # ── 5) info.json 갱신 ───────────────────────────────────────
    info_path = DATASET_ROOT / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["total_tasks"] = len(NEW_TASKS)
    info_path.write_text(json.dumps(info, indent=4))
    print(f"[simplify] info.json: total_tasks={info['total_tasks']}")

    # ── 6) 검증 ─────────────────────────────────────────────────
    print("[simplify] 검증: LeRobotDataset 로딩 및 샘플 task lookup")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id=REPO_ID, root=DATASET_ROOT)
    print(f"  num_episodes={ds.num_episodes}, num_frames={ds.num_frames}, total_tasks={len(ds.meta.tasks)}")
    sample = ds[0]
    print(f"  ds[0]['task_index']={sample['task_index'].item()}  ds[0]['task']={sample['task']!r}")
    full = pd.concat(
        [pd.read_parquet(p) for p in sorted((DATASET_ROOT / "data").rglob("*.parquet"))],
        ignore_index=True,
    )
    print(f"  전체 task_index 분포: {full['task_index'].value_counts().sort_index().to_dict()}")
    print("[simplify] 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
