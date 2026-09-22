"""Dataset scanning + non-destructive edit sidecars for the VLA data editor.

Layout assumed:
    <root>/<dataset>/metadata.json
    <root>/<dataset>/episodes/episode_XXXXXX/{data.hdf5,*.mp4,*.jpg,tasks.json}

All edits live in `edits.json` next to `tasks.json`. Originals are never touched.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

EDITS_VERSION = 1
EDITS_NAME = "edits.json"

# An unrated episode or segment counts as good until a human marks it down.
DEFAULT_SCORE = 1.0
# Back-compat alias: signals.py imports this name.
DEFAULT_SEGMENT_SCORE = DEFAULT_SCORE


# ----------------------------------------------------------------- utilities
def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


# ------------------------------------------------------------------- catalog
class Catalog:
    """Read-only view over the dataset root, plus edit-sidecar read/write."""

    def __init__(self, root: str | Path, history_depth: int = 5,
                 default_vocabulary: list[str] | None = None):
        self.root = Path(root).resolve()
        self.history_depth = int(history_depth)
        self.default_vocabulary = list(default_vocabulary or [])

    # -- paths ------------------------------------------------------------
    def dataset_dir(self, dataset: str) -> Path:
        d = (self.root / dataset).resolve()
        if not d.is_dir() or self.root not in d.parents:
            raise FileNotFoundError(f"unknown dataset: {dataset}")
        return d

    def episode_dir(self, dataset: str, episode: str) -> Path:
        d = (self.dataset_dir(dataset) / "episodes" / episode).resolve()
        if not d.is_dir():
            raise FileNotFoundError(f"unknown episode: {dataset}/{episode}")
        return d

    # -- listing ----------------------------------------------------------
    def list_datasets(self) -> list[dict[str, Any]]:
        out = []
        for child in sorted(self.root.iterdir()):
            meta_path = child / "metadata.json"
            if not meta_path.is_file():
                continue
            meta = _read_json(meta_path)
            eps = self._episode_ids(child)
            labeled = 0
            rejected = 0
            for ep in eps:
                ed = _read_json(child / "episodes" / ep / EDITS_NAME)
                if ed.get("segments"):
                    labeled += 1
                if ed.get("rejected"):
                    rejected += 1
            out.append({
                "name": child.name,
                "robot": meta.get("robot"),
                "collect_hz": meta.get("collect_hz"),
                "cameras": list((meta.get("cameras") or {}).keys()),
                "num_episodes": len(eps),
                "num_labeled": labeled,
                "num_rejected": rejected,
            })
        return out

    @staticmethod
    def _episode_ids(dataset_dir: Path) -> list[str]:
        eps_dir = dataset_dir / "episodes"
        if not eps_dir.is_dir():
            return []
        return sorted(p.name for p in eps_dir.iterdir() if p.is_dir())

    def dataset_detail(self, dataset: str) -> dict[str, Any]:
        ddir = self.dataset_dir(dataset)
        meta = _read_json(ddir / "metadata.json")
        episodes = []
        for ep in self._episode_ids(ddir):
            edir = ddir / "episodes" / ep
            tasks = _read_json(edir / "tasks.json")
            edits = self.load_edits(dataset, ep)
            n = int(tasks.get("num_frames") or 0)
            trim = edits.get("trim") or {}
            start = int(trim.get("start", 0))
            end = int(trim.get("end", n))
            episodes.append({
                "id": ep,
                "index": tasks.get("episode_index"),
                "num_frames": n,
                "kept_frames": max(0, end - start),
                "main_prompt": edits.get("main_prompt", tasks.get("main_prompt", "")),
                "score": _score_or_default(
                    edits.get("score", tasks.get("total_score"))),
                "rejected": bool(edits.get("rejected", False)),
                "num_segments": len(edits.get("segments") or []),
                "updated_at": edits.get("updated_at"),
                "cameras": sorted(p.stem for p in edir.glob("*.mp4")),
            })
        return {
            "name": dataset,
            "metadata": meta,
            "label_vocabulary": self.load_vocabulary(dataset),
            "episodes": episodes,
        }

    # -- vocabulary -------------------------------------------------------
    def vocab_path(self, dataset: str) -> Path:
        return self.dataset_dir(dataset) / "label_vocabulary.json"

    def load_vocabulary(self, dataset: str) -> list[str]:
        """Saved vocabulary, or the configured seed list for a fresh dataset."""
        data = _read_json(self.vocab_path(dataset))
        labels = data.get("labels")
        if isinstance(labels, list):
            return list(labels)
        return list(self.default_vocabulary)

    def save_vocabulary(self, dataset: str, labels: list[str]) -> list[str]:
        clean, seen = [], set()
        for raw in labels:
            label = str(raw).strip()
            if label and label.lower() not in seen:
                seen.add(label.lower())
                clean.append(label)
        _write_json_atomic(self.vocab_path(dataset), {"labels": clean})
        return clean

    # -- edits sidecar ----------------------------------------------------
    def edits_path(self, dataset: str, episode: str) -> Path:
        return self.episode_dir(dataset, episode) / EDITS_NAME

    def load_edits(self, dataset: str, episode: str) -> dict[str, Any]:
        try:
            return _read_json(self.edits_path(dataset, episode))
        except FileNotFoundError:
            return {}

    def default_edits(self, dataset: str, episode: str) -> dict[str, Any]:
        """Edits seeded from the original tasks.json, used when none exist yet."""
        tasks = _read_json(self.episode_dir(dataset, episode) / "tasks.json")
        n = int(tasks.get("num_frames") or 0)
        return {
            "version": EDITS_VERSION,
            "episode_id": episode,
            "main_prompt": tasks.get("main_prompt", ""),
            "score": _score_or_default(tasks.get("total_score")),
            "rejected": False,
            "notes": "",
            "trim": {"start": 0, "end": n},
            "segments": [],
            "updated_at": None,
        }

    def get_edits(self, dataset: str, episode: str) -> dict[str, Any]:
        existing = self.load_edits(dataset, episode)
        merged = self.default_edits(dataset, episode)
        merged.update(existing)
        return merged

    def save_edits(self, dataset: str, episode: str, payload: dict[str, Any]) -> dict[str, Any]:
        current = self.get_edits(dataset, episode)
        n = int(_read_json(self.episode_dir(dataset, episode) / "tasks.json").get("num_frames") or 0)

        for key in ("main_prompt", "notes"):
            if key in payload:
                current[key] = str(payload[key] or "")
        if "score" in payload:
            current["score"] = _score_or_default(payload["score"])
        if "rejected" in payload:
            current["rejected"] = bool(payload["rejected"])
        if "trim" in payload:
            current["trim"] = _sanitize_trim(payload["trim"], n)
        if "segments" in payload:
            current["segments"] = _sanitize_segments(payload["segments"], n)

        current["version"] = EDITS_VERSION
        current["episode_id"] = episode
        current["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self._backup(self.edits_path(dataset, episode), keep=self.history_depth)
        _write_json_atomic(self.edits_path(dataset, episode), current)
        return current

    @staticmethod
    def _backup(path: Path, keep: int = 5) -> None:
        """Keep a short rotating history so a bad save is recoverable."""
        if keep <= 0 or not path.is_file():
            return
        hist = path.parent / ".edits_history"
        hist.mkdir(exist_ok=True)
        shutil.copy2(path, hist / f"{time.strftime('%Y%m%d-%H%M%S')}.json")
        snaps = sorted(hist.glob("*.json"))
        for old in snaps[:-keep]:
            old.unlink(missing_ok=True)


# --------------------------------------------------------------- validation
def _clamp_opt_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def _score_or_default(value: Any) -> float:
    """Score with the unset case folded to DEFAULT_SCORE.

    Used for both the episode and its segments: an unrated item counts as good
    until a human marks it down, so only a downgrade needs typing.
    """
    score = _clamp_opt_float(value)
    return DEFAULT_SCORE if score is None else score


def _sanitize_trim(trim: Any, num_frames: int) -> dict[str, int]:
    trim = trim or {}
    start = max(0, min(int(trim.get("start", 0) or 0), num_frames))
    end = max(start, min(int(trim.get("end", num_frames) or num_frames), num_frames))
    return {"start": start, "end": end}


def _sanitize_segments(segments: Any, num_frames: int) -> list[dict[str, Any]]:
    """Clamp to range, drop empties, sort by start. Overlap is allowed but flagged by the UI.

    An unscored segment defaults to DEFAULT_SCORE: a labelled subtask is
    assumed good unless a human marks it down, so the common case needs no
    typing.
    """
    out = []
    for seg in segments or []:
        try:
            start = max(0, min(int(seg.get("start", 0)), num_frames))
            end = max(0, min(int(seg.get("end", 0)), num_frames))
        except (TypeError, ValueError, AttributeError):
            continue
        if end <= start:
            continue
        out.append({
            "start": start,
            "end": end,
            "label": str(seg.get("label", "") or "").strip(),
            "prompt": str(seg.get("prompt", "") or "").strip(),
            "score": _score_or_default(seg.get("score")),
        })
    out.sort(key=lambda s: (s["start"], s["end"]))
    return out
