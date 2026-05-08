"""로컬 LeRobot v3.0 데이터셋의 stats 를 data parquet 으로부터 재계산.

`data_conversion.py` 가 로컬 모드에서 stats 재계산을 건너뛰는 버그가 있어,
변환된 데이터셋의 per-episode stats 와 global stats.json 이 옛 표현(8D joint)
그대로 남아 있는 문제를 해결한다.

처리 대상:
  1) meta/episodes/**/*.parquet 의 `stats/{feature}/{metric}` 컬럼 전체 재계산
     - feature = data parquet 의 모든 numeric/array 컬럼
     - metric  = min, max, mean, std, count, q01, q10, q50, q90, q99
  2) meta/stats.json 의 모든 feature 통계 재계산 (=글로벌 통계)

이미지 / 비디오 통계는 이 스크립트가 손대지 않는다 (별도 디코딩 필요).
원본 stats.json 의 image/video 항목은 그대로 유지한다.

사용:
  python scripts/data_fix_stats.py <dataset_root> [<dataset_root> ...]

예:
  python scripts/data_fix_stats.py \
      /workspace/m.ax/datasets/manufacturing_kitting_dataset \
      /workspace/m.ax/datasets/manufacturing_kitting_dataset_flip_object \
      /workspace/m.ax/datasets/manufacturing_kitting_dataset_kit_object
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_STATS_METRICS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
_QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)
_QUANTILE_KEYS = ("q01", "q10", "q50", "q90", "q99")
_NON_FEATURE_COLUMNS_PASS = ("timestamp", "frame_index", "episode_index", "index", "task_index")


def _is_array_like(v) -> bool:
    return isinstance(v, (list, np.ndarray))


def _stack_column(series: pd.Series) -> np.ndarray:
    """1차원 (N,) 또는 2차원 (N, D) numpy 배열로 정규화."""
    sample = series.iloc[0]
    if _is_array_like(sample):
        return np.stack([np.asarray(v, dtype=np.float64) for v in series.to_numpy()])
    return series.to_numpy().astype(np.float64)


def _compute_stats(arr: np.ndarray) -> dict[str, np.ndarray]:
    """1D 또는 2D 배열에 대한 lerobot 표준 통계.

    - 1D (N,)   : 결과는 scalar 들을 (1,) 배열로 감싸 저장 (lerobot 관례)
    - 2D (N, D) : 결과는 (D,) 배열
    """
    if arr.ndim == 1:
        out = {
            "min":   np.array([float(arr.min())]),
            "max":   np.array([float(arr.max())]),
            "mean":  np.array([float(arr.mean())]),
            "std":   np.array([float(arr.std())]),
            "count": np.array([int(arr.size)]),
        }
        for q, qk in zip(_QUANTILES, _QUANTILE_KEYS):
            out[qk] = np.array([float(np.quantile(arr, q))])
        return out
    # 2D
    out = {
        "min":   arr.min(axis=0).astype(np.float64),
        "max":   arr.max(axis=0).astype(np.float64),
        "mean":  arr.mean(axis=0).astype(np.float64),
        "std":   arr.std(axis=0).astype(np.float64),
        "count": np.array([int(arr.shape[0])]),
    }
    for q, qk in zip(_QUANTILES, _QUANTILE_KEYS):
        out[qk] = np.quantile(arr, q, axis=0).astype(np.float64)
    return out


def _to_serializable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def fix_one(root: Path) -> None:
    print(f"\n{'='*78}")
    print(f"FIX: {root}")
    print(f"{'='*78}")

    info_path = root / "meta" / "info.json"
    stats_path = root / "meta" / "stats.json"
    info = json.loads(info_path.read_text())
    declared_features = set(info["features"].keys())
    image_video_features = {
        k for k, v in info["features"].items() if v.get("dtype") in ("image", "video")
    }
    print(f"  info.json features: {len(declared_features)} (image/video: {len(image_video_features)})")

    # ── 1) data parquet 로딩 + 컬럼별 통계 재계산 (이미지/비디오 제외) ──
    data_files = sorted(root.glob("data/**/*.parquet"))
    if not data_files:
        raise SystemExit(f"data parquet 없음: {root}")
    df = pd.concat([pd.read_parquet(f) for f in data_files], ignore_index=True)
    print(f"  data frames: {len(df)}  episodes: {df['episode_index'].nunique()}")

    numeric_cols = [
        c for c in df.columns
        if c not in image_video_features  # 이미지/비디오는 스킵
    ]

    # ── 2) global stats 재계산 ──────────────────────────────────────
    print(f"  ── global stats 재계산 ({len(numeric_cols)} numeric features) ──")
    new_global = {}
    for col in numeric_cols:
        arr = _stack_column(df[col])
        new_global[col] = _compute_stats(arr)
        dim = "scalar" if arr.ndim == 1 else f"D={arr.shape[1]}"
        print(f"    {col:50s} {dim}  count={arr.shape[0] if arr.ndim==1 else arr.shape[0]}")

    # ── 3) 이미지/비디오 stats 는 기존 값 보존 ──────────────────────
    if stats_path.exists():
        old_stats = json.loads(stats_path.read_text())
        for k in image_video_features:
            if k in old_stats:
                new_global[k] = old_stats[k]
                print(f"    {k:50s} 이미지/비디오 stats 보존")

    # ── 4) declared features 와 일치하는지 검증 ─────────────────────
    final_keys = set(new_global.keys())
    missing_in_data = declared_features - final_keys
    extra_in_data = final_keys - declared_features
    if missing_in_data:
        print(f"  ⚠ info.json 에 선언됐지만 stats 못 만든 features: {sorted(missing_in_data)}")
    if extra_in_data:
        print(f"  ⚠ data 엔 있지만 info.json 미선언 (stats 만 만들어짐): {sorted(extra_in_data)}")

    # ── 5) global stats.json 저장 (이전을 .bak.json 로 백업) ─────────
    if stats_path.exists():
        bak = stats_path.with_suffix(".bak.json")
        if bak.exists():
            bak2 = stats_path.with_suffix(".bak2.json")
            bak.rename(bak2)
            print(f"  기존 백업 보존: stats.bak.json → stats.bak2.json")
        stats_path.rename(bak)
        print(f"  새 백업 생성: stats.json → stats.bak.json")

    serializable = {k: {m: _to_serializable(v) for m, v in fs.items()} for k, fs in new_global.items()}
    stats_path.write_text(json.dumps(serializable, indent=4))
    print(f"  ✅ stats.json 갱신 ({len(new_global)} features)")

    # ── 6) per-episode stats 재계산 (모든 numeric feature, 이미지/비디오 제외) ──
    ep_files = sorted(root.glob("meta/episodes/**/*.parquet"))
    if not ep_files:
        print("  ⚠ episode meta 없음 - skip per-episode")
        return
    print(f"  ── per-episode stats 재계산 ({len(ep_files)} files) ──")

    # 각 ep 별로 stats 미리 계산
    ep_to_stats: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for ep, g in df.groupby("episode_index"):
        ep_stats = {}
        for col in numeric_cols:
            arr = _stack_column(g[col])
            ep_stats[col] = _compute_stats(arr)
        ep_to_stats[int(ep)] = ep_stats

    for ep_path in ep_files:
        ep_df = pd.read_parquet(ep_path)
        # 기존 stats/* 컬럼 모두 제거하고 새로 채움 (옛 8D 잔재 제거)
        old_stat_cols = [c for c in ep_df.columns if c.startswith("stats/")]
        # 이미지/비디오 stats 컬럼은 보존 (이 스크립트가 안 만드므로)
        keep = [c for c in old_stat_cols if c.split("/")[1] in image_video_features]
        drop = [c for c in old_stat_cols if c not in keep]
        ep_df = ep_df.drop(columns=drop)

        ep_indices = ep_df["episode_index"].tolist()
        # 새 stats 컬럼들을 한 번에 모아서 concat (단편화 방지)
        new_stat_cols: dict[str, list] = {}
        for col in numeric_cols:
            for metric in _STATS_METRICS:
                key = f"stats/{col}/{metric}"
                values = []
                for ep in ep_indices:
                    v = ep_to_stats[int(ep)][col][metric]
                    if isinstance(v, np.ndarray):
                        values.append(v.copy())
                    else:
                        values.append(np.array([v]))
                new_stat_cols[key] = values
        new_stats_df = pd.DataFrame(new_stat_cols, index=ep_df.index)
        ep_df = pd.concat([ep_df, new_stats_df], axis=1)

        ep_df.to_parquet(ep_path, index=False)
        print(f"    {ep_path.relative_to(root)}: 갱신 (keep={len(keep)}, replace={len(drop)}, add={len(new_stat_cols)})")

    print(f"  ✅ per-episode stats 갱신 완료")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="+", type=Path, help="LeRobot v3.0 dataset root 디렉토리들")
    args = parser.parse_args()

    for root in args.roots:
        if not (root / "meta" / "info.json").exists():
            raise SystemExit(f"meta/info.json 없음: {root}")
        fix_one(root)

    print("\n[fix_stats] 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
