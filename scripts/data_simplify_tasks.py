"""tasks.parquet 을 4개 → 2개로 축소.

기존 4개 prompt (flip part / flip object / kit object / kit part) 를 part/object
구분 없이 동작 단위 2개로 합쳐 instruction-conditioning 을 단순화한다:

  0  flip object   ← flip part / flip object
  1  kit object    ← kit object / kit part

수정 대상:
  - meta/tasks.parquet         : 2행으로 재작성
  - data/**/*.parquet          : task_index 컬럼 전체 리매핑
  - meta/episodes/**/*.parquet : tasks 컬럼(에피소드별 paraphrase 리스트) 재계산
                               + stats/task_index/* 컬럼 재계산
  - meta/info.json             : total_tasks=2
  - meta/stats.json            : task_index 항목 재계산

기존 .bak.parquet 백업본은 .bak2.parquet 로 옮긴 뒤 새 백업을 .bak.parquet 로 저장.
멱등성 없음 (한 번 실행하면 옛 4-task 인덱스는 사라짐).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "thirdparty" / "lerobot" / "src"))

import numpy as np
import pandas as pd

REPO_ID = "manufacturing_kitting_dataset"
DATASET_ROOT = Path("/workspace/m.ax/datasets") / REPO_ID

NEW_TASKS = ["flip object", "kit object"]
NEW_NAME_TO_IDX = {name: i for i, name in enumerate(NEW_TASKS)}

# 기존 4개 prompt → 새 인덱스 (0=flip object, 1=kit object)
OLD_NAME_TO_NEW_IDX: dict[str, int] = {
    "flip part":   0,
    "flip object": 0,
    "kit object":  1,
    "kit part":    1,
}

# stats 항목 (lerobot v3.0 표준)
_STATS_METRICS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")


def _compute_1d_stats(arr: np.ndarray) -> dict[str, float]:
    """1차원 numeric 배열에 대한 lerobot v3.0 표준 통계 dict 반환."""
    arr = np.asarray(arr, dtype=np.float64).ravel()
    return {
        "min":   float(arr.min()),
        "max":   float(arr.max()),
        "mean":  float(arr.mean()),
        "std":   float(arr.std()),
        "count": int(arr.size),
        "q01":   float(np.quantile(arr, 0.01)),
        "q10":   float(np.quantile(arr, 0.10)),
        "q50":   float(np.quantile(arr, 0.50)),
        "q90":   float(np.quantile(arr, 0.90)),
        "q99":   float(np.quantile(arr, 0.99)),
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

    # ── 3) episodes 메타의 tasks 컬럼 + stats/task_index/* 재계산 ──
    # tasks 컬럼 = 해당 에피소드에 등장하는 unique paraphrase 리스트.
    # 새 task_index 분포로부터 다시 계산.
    ep_to_new_paraphrases: dict[int, list[str]] = {}
    ep_to_task_stats: dict[int, dict[str, np.ndarray]] = {}
    for parquet_path in sorted(data_dir.rglob("*.parquet")):
        df = pd.read_parquet(parquet_path, columns=["episode_index", "task_index"])
        for ep, g in df.groupby("episode_index"):
            uniq_new_idx = sorted(int(i) for i in g["task_index"].unique())
            ep_to_new_paraphrases[int(ep)] = [NEW_TASKS[i] for i in uniq_new_idx]
            arr = g["task_index"].to_numpy().astype(np.float64)
            ep_to_task_stats[int(ep)] = _compute_1d_stats(arr)

    episodes_dir = DATASET_ROOT / "meta" / "episodes"
    for parquet_path in sorted(episodes_dir.rglob("*.parquet")):
        ep_df = pd.read_parquet(parquet_path)
        new_col = [ep_to_new_paraphrases[int(ep)] for ep in ep_df["episode_index"].tolist()]
        ep_df["tasks"] = new_col
        # stats/task_index/{metric} 컬럼 갱신 (있는 경우만)
        for metric in _STATS_METRICS:
            col = f"stats/task_index/{metric}"
            if col in ep_df.columns:
                ep_df[col] = [
                    np.array([ep_to_task_stats[int(ep)][metric]])
                    for ep in ep_df["episode_index"].tolist()
                ]
        ep_df.to_parquet(parquet_path, index=False)
        eps = ep_df["episode_index"]
        print(f"  meta: {parquet_path.relative_to(DATASET_ROOT)}  "
              f"ep{int(eps.min())}-{int(eps.max())} tasks/stats 갱신")

    # ── 4) tasks.parquet 재작성 (백업 후) ────────────────────────
    bak_path = tasks_path.with_suffix(".bak.parquet")
    if bak_path.exists():
        bak2_path = tasks_path.with_suffix(".bak2.parquet")
        shutil.move(str(bak_path), str(bak2_path))
        print(f"[simplify] 기존 백업 보존: {bak_path.name} → {bak2_path.name}")
    shutil.copy2(tasks_path, bak_path)
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

    # ── 6) stats.json 의 task_index 항목 재계산 ─────────────────
    # 다른 feature 통계는 trim 시점에 이미 재계산되어 정확하므로 건드리지 않음.
    stats_path = DATASET_ROOT / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    full_task_idx = pd.concat(
        [pd.read_parquet(p, columns=["task_index"])
         for p in sorted((DATASET_ROOT / "data").rglob("*.parquet"))],
        ignore_index=True,
    )["task_index"].to_numpy()
    new_task_stats = _compute_1d_stats(full_task_idx)
    # lerobot v3.0 stats.json 은 1-원소 list 로 저장
    stats["task_index"] = {m: [new_task_stats[m]] for m in _STATS_METRICS}
    stats_path.write_text(json.dumps(stats, indent=4))
    print(f"[simplify] stats.json: task_index 재계산 "
          f"(min={new_task_stats['min']:.0f} max={new_task_stats['max']:.0f} "
          f"mean={new_task_stats['mean']:.4f} count={new_task_stats['count']})")

    # ── 7) 검증 ─────────────────────────────────────────────────
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
    dist = full["task_index"].value_counts().sort_index().to_dict()
    print(f"  전체 task_index 분포: {dist}")
    expected = set(range(len(NEW_TASKS)))
    actual = set(int(k) for k in dist.keys())
    if not actual.issubset(expected):
        raise SystemExit(f"task_index 분포 이상: 기대={expected}, 실제={actual}")
    # 에피소드별 tasks 리스트가 단일 원소인지 확인 (한 에피소드는 한 task 만 가짐)
    ep_full = pd.concat(
        [pd.read_parquet(p) for p in sorted((DATASET_ROOT / "meta" / "episodes").rglob("*.parquet"))],
        ignore_index=True,
    )
    ep_task_lens = [len(t) for t in ep_full["tasks"]]
    print(f"  episodes tasks 리스트 길이 분포: min={min(ep_task_lens)} max={max(ep_task_lens)} "
          f"(전체 {len(ep_full)} 에피소드)")
    if max(ep_task_lens) != 1:
        print(f"  ⚠ 일부 에피소드가 둘 이상의 task 를 포함함 (의도치 않은 분기 가능성)")
    # stats.json 의 task_index 가 새 분포 [0..len(NEW_TASKS)-1] 와 일치하는지
    saved = json.loads(stats_path.read_text())["task_index"]
    saved_max = saved["max"][0] if isinstance(saved["max"], list) else saved["max"]
    if int(saved_max) != len(NEW_TASKS) - 1:
        raise SystemExit(f"stats.json task_index.max={saved_max} 가 기대값 {len(NEW_TASKS)-1} 와 다름")
    print(f"  stats.json task_index ✅ (min=0, max={int(saved_max)})")
    print("[simplify] 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
