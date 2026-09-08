#!/usr/bin/env python3
"""Merge several raw max_data_collect datasets into one, by renumbering.

Physical merge only: HDF5 signal arrays and mp4 pixels are never touched,
re-encoded or resampled. The only thing that changes is an episode's
number and the places that record it -- which is what makes datasets
collected on different days combinable at all, since each one starts
numbering near 0 and they collide wholesale.

The episode number lives in FOUR places, and every one of them has to
move together or the dataset is quietly inconsistent:

    1. the directory name        episodes/episode_NNNNNN/
    2. tasks.json                episode_id, episode_index
    3. data.hdf5 /meta attrs     episode_id, episode_index
    4. metadata.json             episodes[].id, episodes[].index

Sources are merged in the order listed in `merge_target`, and within a
source in ascending episode number. Output numbering is gapless from 0,
so a gap left by a deleted episode in a source does not propagate.

Per-episode prompts ride along in tasks.json untouched, which is the
point: merging flip/pick/kit datasets yields one multi-task dataset.
Because two sources can share a prompt while differing in how they were
collected, every merged episode also records where it came from
(`source_dataset` / `source_episode_id`) -- that is the only trace of
provenance once the numbers are gone, and it makes the merge reversible.

Usage:
    edit merge_target and OUTPUT below, then

    python3 merge_raw_episode.py --dry-run     # show the plan, touch nothing
    python3 merge_raw_episode.py               # do it (hardlinks, ~instant)

Sources are opened read-only unless --move is given.
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py


# ── what to merge ──────────────────────────────────────────────────────
# Edit this list. Order matters: it is the order episodes are numbered
# in. Paths may be absolute or relative to DATASETS_ROOT.

DATASETS_ROOT = Path("/workspace/m.ax/datasets")

merge_target = [
    "kitting_dataset_xarm7_pick_260825",
    "kitting_dataset_xarm7_flip_260825",
    "kitting_dataset_xarm7_kit_260825",
    "kitting_dataset_xarm7_kit_260826",
    "kitting_dataset_xarm7_pick_kit_260825",
    "kitting_dataset_xarm7_flip_pick_kit_260826",
    "kitting_dataset_xarm7_random_slot_kit_260826",
]

OUTPUT = "kitting_dataset_xarm7_merged"


# ── compatibility ──────────────────────────────────────────────────────
# Fields that must agree across sources for a merge to mean anything. A
# mismatch here is not cosmetic: mixing 30 Hz with 60 Hz episodes, or
# datasets whose joint order differs, produces a dataset whose rows do
# not mean the same thing -- so it is refused rather than warned about
# (--force downgrades it to a warning).
COMPATIBILITY_FIELDS = (
    "collect_hz", "robot", "gripper", "joint_names", "cameras",
    "video_codec", "features",
)

EPISODE_PREFIX = "episode_"


def log(message: str) -> None:
    print(message, flush=True)


# ── reading sources ────────────────────────────────────────────────────

def load_metadata(root: Path) -> dict:
    path = root / "metadata.json"
    if not path.exists():
        raise SystemExit(f"error: {path} does not exist -- not a dataset")
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"error: cannot read {path}: {exc}")


def scan_episodes(root: Path) -> list[tuple[int, Path]]:
    """Episode dirs that actually hold data, sorted by number.

    A directory with no data.hdf5 is left over from an interrupted
    recording; the recorder never reuses its number and neither do we.
    """
    episodes_dir = root / "episodes"
    if not episodes_dir.is_dir():
        raise SystemExit(f"error: {episodes_dir} does not exist")

    found = []
    for path in sorted(episodes_dir.iterdir()):
        if not path.is_dir() or not path.name.startswith(EPISODE_PREFIX):
            continue
        suffix = path.name[len(EPISODE_PREFIX):]
        if not suffix.isdigit():
            continue
        if not (path / "data.hdf5").exists():
            log(f"  ! skipping {root.name}/{path.name}: no data.hdf5 "
                "(interrupted recording)")
            continue
        found.append((int(suffix), path))
    return found


def check_compatible(sources: list[tuple[Path, dict]], force: bool) -> None:
    """Refuse to merge datasets that do not describe the same setup."""
    reference_root, reference = sources[0]
    problems = []
    for root, meta in sources[1:]:
        for field in COMPATIBILITY_FIELDS:
            if meta.get(field) != reference.get(field):
                problems.append(
                    f"  {field}: {reference_root.name}="
                    f"{json.dumps(reference.get(field))[:80]} "
                    f"vs {root.name}={json.dumps(meta.get(field))[:80]}"
                )
    if not problems:
        return
    header = (f"sources disagree on {len(problems)} field(s):\n"
              + "\n".join(problems))
    if not force:
        raise SystemExit(
            f"error: {header}\n"
            "merging these would produce a dataset whose rows do not mean "
            "the same thing. Pass --force to merge anyway."
        )
    log(f"WARNING: {header}\n  --force given, merging anyway")


# ── writing the merge ──────────────────────────────────────────────────

def place_file(src: Path, dst: Path, mode: str) -> None:
    """Put one file in place by the cheapest means the mode allows.

    Hardlinking 1.8 GB of video costs no time and no disk, but it means
    src and dst are the SAME inode -- editing one edits both. Only files
    we never rewrite may be linked; see place_episode.
    """
    if mode == "move":
        shutil.move(str(src), str(dst))
        return
    if mode == "link":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass                # different filesystem; fall through
    shutil.copy2(src, dst)


def place_episode(src_dir: Path, dst_dir: Path, index: int,
                  dataset_name: str, mode: str) -> dict:
    """Copy/link one episode into place and renumber it.

    Returns the metadata.json entry for the merged episode.
    """
    episode_id = f"{EPISODE_PREFIX}{index:06d}"
    dst_dir.mkdir(parents=True)

    for item in sorted(src_dir.iterdir()):
        if not item.is_file():
            continue
        # data.hdf5 gets its /meta attrs rewritten below, so it must NOT
        # share an inode with the source -- a hardlink here would edit
        # the original dataset's file too. Everything else (mp4, jpg) is
        # byte-identical in the merge and can be linked freely.
        if item.name == "data.hdf5":
            place_file(item, dst_dir / item.name,
                       "move" if mode == "move" else "copy")
        elif item.name == "tasks.json":
            pass                # rewritten wholesale below
        else:
            place_file(item, dst_dir / item.name, mode)

    rewrite_hdf5(dst_dir / "data.hdf5", episode_id, index)
    entry = rewrite_tasks(
        src_dir / "tasks.json", dst_dir / "tasks.json", episode_id, index,
        dataset_name, src_dir.name,
    )
    if mode == "move":
        shutil.rmtree(src_dir, ignore_errors=True)
    return entry


def rewrite_hdf5(path: Path, episode_id: str, index: int) -> None:
    """Renumber the HDF5 in place -- attrs only, arrays untouched.

    Opening 'r+' rewrites two small attributes and leaves every dataset
    byte-for-byte as recorded. This is the step most easily forgotten,
    and an episode whose /meta disagrees with its directory name is the
    kind of bug that only shows up much later, in training.
    """
    try:
        with h5py.File(path, "r+") as f:
            meta = f["meta"]
            meta.attrs["episode_id"] = episode_id
            meta.attrs["episode_index"] = index
    except (OSError, KeyError) as exc:
        raise SystemExit(f"error: cannot renumber {path}: {exc}")


def rewrite_tasks(src: Path, dst: Path, episode_id: str, index: int,
                  dataset_name: str, source_episode_id: str) -> dict:
    """Renumber tasks.json, keeping the prompt and adding provenance."""
    task = {}
    if src.exists():
        try:
            task = json.loads(src.read_text())
        except (OSError, ValueError) as exc:
            log(f"  ! {src}: unreadable ({exc}); writing a minimal tasks.json")
    else:
        log(f"  ! {src}: missing; writing a minimal tasks.json")

    task["episode_id"] = episode_id
    task["episode_index"] = index
    # Where this episode came from. Two sources can share a prompt while
    # differing in setup, so the prompt alone cannot identify the origin.
    task["source_dataset"] = dataset_name
    task["source_episode_id"] = source_episode_id
    dst.write_text(json.dumps(task, indent=2, ensure_ascii=False) + "\n")

    return {
        "id": episode_id,
        "index": index,
        "num_frames": task.get("num_frames", 0),
        "duration": task.get("duration", 0.0),
        "source": task.get("source", ""),
        "dropped": task.get("dropped", 0),
        "source_dataset": dataset_name,
        "source_episode_id": source_episode_id,
    }


def merge_entry(entry: dict, source_meta_entry: dict | None) -> dict:
    """Prefer the source metadata.json's numbers where tasks.json lacks them.

    duration lives only in metadata.json, so a merge driven purely by
    tasks.json would silently lose it.
    """
    if source_meta_entry:
        for field in ("num_frames", "duration", "source", "dropped"):
            if field in source_meta_entry:
                entry[field] = source_meta_entry[field]
    return entry


# ── main ───────────────────────────────────────────────────────────────

def resolve(name: str) -> Path:
    path = Path(name)
    return path if path.is_absolute() else DATASETS_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge raw max_data_collect datasets by renumbering "
                    "episodes. Edit merge_target at the top of this file.",
    )
    parser.add_argument(
        "--out", default=None,
        help=f"output dataset path or name (default: {OUTPUT})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the plan and exit without touching anything",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--copy", action="store_true",
        help="copy files instead of hardlinking (slower, uses disk, but "
             "the merge is fully independent of the sources)",
    )
    group.add_argument(
        "--move", action="store_true",
        help="move files out of the sources, consuming them",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="add to an existing output dataset instead of refusing",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="merge even if the sources disagree on hz/robot/cameras/...",
    )
    args = parser.parse_args()

    if not merge_target:
        raise SystemExit(
            "error: merge_target is empty -- edit the list at the top of "
            f"{Path(__file__).name}"
        )

    mode = "copy" if args.copy else "move" if args.move else "link"
    out_root = resolve(args.out or OUTPUT)

    # ── gather and validate before writing anything ────────────────────
    roots = [resolve(name) for name in merge_target]
    duplicates = {r for r in roots if roots.count(r) > 1}
    if duplicates:
        raise SystemExit(
            f"error: merge_target lists the same dataset twice: "
            f"{sorted(d.name for d in duplicates)}"
        )
    if out_root in roots:
        raise SystemExit(
            f"error: output {out_root} is also a source; that would consume "
            "the dataset it is writing into"
        )

    sources = []
    for root in roots:
        if not root.is_dir():
            raise SystemExit(f"error: {root} does not exist")
        sources.append((root, load_metadata(root)))
    check_compatible(sources, args.force)

    plans, index = [], 0
    existing_entries = []
    if out_root.exists():
        if not args.append:
            raise SystemExit(
                f"error: {out_root} already exists. Remove it, choose "
                "another --out, or pass --append to add to it."
            )
        existing = load_metadata(out_root)
        existing_entries = existing.get("episodes", [])
        numbers = [n for n, _ in scan_episodes(out_root)]
        index = (max(numbers) + 1) if numbers else 0
        log(f"appending to {out_root} ({len(numbers)} episode(s) present, "
            f"continuing at {index})")

    total_episodes = 0
    for root, meta in sources:
        episodes = scan_episodes(root)
        by_id = {e.get("id"): e for e in meta.get("episodes", [])}
        first = index
        for number, path in episodes:
            plans.append((path, index, root.name, by_id.get(path.name)))
            index += 1
        total_episodes += len(episodes)
        if episodes:
            log(f"  {root.name}: {len(episodes)} episode(s) "
                f"-> {EPISODE_PREFIX}{first:06d}..{EPISODE_PREFIX}{index - 1:06d}")
        else:
            log(f"  {root.name}: no episodes, skipped")

    if not plans:
        raise SystemExit("error: the sources contain no episodes")

    log(f"\n{total_episodes} episode(s) -> {out_root} (mode: {mode})")
    if args.dry_run:
        log("dry run: nothing written")
        return 0

    # ── write ──────────────────────────────────────────────────────────
    episodes_dir = out_root / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    entries = list(existing_entries)
    for i, (src_dir, number, dataset_name, meta_entry) in enumerate(plans, 1):
        dst_dir = episodes_dir / f"{EPISODE_PREFIX}{number:06d}"
        entry = place_episode(src_dir, dst_dir, number, dataset_name, mode)
        entries.append(merge_entry(entry, meta_entry))
        if i % 20 == 0 or i == len(plans):
            log(f"  {i}/{len(plans)} episodes")

    # Dataset-level shell: the sources agree on all of it (or --force was
    # given and the first one wins), so it is copied from the first
    # source with only the name and episode list replaced.
    metadata = dict(sources[0][1])
    metadata["dataset_name"] = out_root.name
    metadata["episodes"] = entries
    metadata["merged_from"] = [root.name for root, _ in sources]
    metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

    tmp = out_root / "metadata.json.tmp"
    tmp.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(out_root / "metadata.json")

    log(f"\ndone: {len(entries)} episode(s) in {out_root}")
    if mode == "link":
        log("videos are hardlinked to the sources -- deleting a source "
            "directory does not affect them, but editing a video in place "
            "would change both.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
