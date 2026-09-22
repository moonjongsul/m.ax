#!/usr/bin/env python3
"""Upload a converted LeRobot dataset (see convert_to_lerobot.py) to the Hub.

This is a thin wrapper around `LeRobotDataset.push_to_hub`, which already does
the right thing: it creates the repo, uploads `root` as-is, writes a dataset
card from `meta/info.json`, and tags the revision with the codebase version.
What this script adds is the checking that has to happen BEFORE a 2.5 GB
upload starts, and a default that suits this dataset:

    * `--private` by default. Robot data is not obviously publishable, and a
      dataset pushed public cannot be un-published from anyone who already
      pulled it. Making it public has to be typed out (`--public`).
    * The subtask metadata is verified to be present and non-empty. The whole
      point of this dataset is the editor's segment labels; a dataset whose
      `subtask_index` is -1 everywhere is the pre-fix conversion, and pushing
      it wastes an upload and, worse, looks correct on the Hub.
    * `upload_large_folder` is the default transfer mode. It is resumable and
      multi-threaded, which for ~2.5 GB of video over a flaky link is the
      difference between "retry the failed shards" and "start over".

Note on `meta/subtasks.parquet`: LeRobot 0.5.1 reads that file but has no
writer for it, so the converter writes it. `push_to_hub` passes no
`allow_patterns`, uploading everything but `images/`, so it rides along --
this script asserts it actually arrived rather than assuming it.

Usage:
    python3 upload_to_hub.py --root datasets/lerobot/xarm7_kitting_v3 \
                             --repo-id <user>/<name>            # private
    python3 upload_to_hub.py --root ... --repo-id ... --dry-run # check only
    python3 upload_to_hub.py --root ... --repo-id ... --public

Authentication is huggingface_hub's own: `hf auth login`, or HF_TOKEN in the
environment. This script never takes a token as an argument, so it cannot end
up in your shell history.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ───────────────────────────────────────────────────────────── inspection
def dataset_summary(root: Path) -> dict:
    """What is in this dataset, read straight off the metadata files.

    Deliberately does not construct a `LeRobotDataset`: that wants to resolve
    video backends and would fail on a machine that can merely upload.
    """
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise SystemExit(f"not a LeRobot dataset (no meta/info.json): {root}")
    info = json.loads(info_path.read_text(encoding="utf-8"))

    summary = {
        "codebase_version": info.get("codebase_version", "?"),
        "robot_type": info.get("robot_type", "?"),
        "episodes": info.get("total_episodes", 0),
        "frames": info.get("total_frames", 0),
        "tasks": info.get("total_tasks", 0),
        "fps": info.get("fps", 0),
        "features": sorted(info.get("features", {})),
        "size_mb": sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) / 1e6,
    }

    import pandas as pd

    subtasks_path = root / "meta" / "subtasks.parquet"
    if subtasks_path.is_file():
        frame = pd.read_parquet(subtasks_path)
        # The trailing sentinel row (index -1, empty name) is bookkeeping for
        # the -1 "no subtask" marker, not a real subtask -- don't count it.
        summary["subtask_prompts"] = [str(name) for name in frame.index if str(name).strip()]
    else:
        summary["subtask_prompts"] = []

    annotated = 0
    total = 0
    for path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        frame = pd.read_parquet(path)
        total += len(frame)
        if "sparse_subtask_names" in frame.columns:
            annotated += int(frame["sparse_subtask_names"].notna().sum())
    summary["episodes_annotated"] = annotated
    summary["episodes_in_meta"] = total
    return summary


def check_subtasks(summary: dict) -> list[str]:
    """Reasons this dataset looks like it lost its subtask labels."""
    problems = []
    if not summary["subtask_prompts"]:
        problems.append("meta/subtasks.parquet is missing or empty "
                        "(no sentence prompts to condition a policy on)")
    if summary["episodes_annotated"] == 0:
        problems.append("no episode carries sparse_subtask_names "
                        "(the visualizer timeline and SARM will see nothing)")
    elif summary["episodes_annotated"] < summary["episodes_in_meta"]:
        missing = summary["episodes_in_meta"] - summary["episodes_annotated"]
        problems.append(f"{missing} of {summary['episodes_in_meta']} episodes "
                        "have no sparse_subtask_names")
    return problems


def print_summary(root: Path, repo_id: str, summary: dict, private: bool) -> None:
    print(f"root           : {root}")
    print(f"repo_id        : {repo_id}   ({'private' if private else 'PUBLIC'})")
    print(f"codebase       : {summary['codebase_version']}   robot: {summary['robot_type']}")
    print(f"episodes       : {summary['episodes']}   frames: {summary['frames']}   "
          f"fps: {summary['fps']}")
    print(f"tasks          : {summary['tasks']}")
    print(f"upload size    : {summary['size_mb']:.0f} MB")
    print(f"annotated eps  : {summary['episodes_annotated']}/{summary['episodes_in_meta']}")
    print(f"subtasks       : {len(summary['subtask_prompts'])}")
    for prompt in summary["subtask_prompts"]:
        print(f"  - {prompt}")
    print("features:")
    for name in summary["features"]:
        print(f"  {name}")


# ──────────────────────────────────────────────────────────────── upload
def upload(root: Path, repo_id: str, private: bool, tags: list[str],
           license_: str, branch: str | None, push_videos: bool,
           large: bool) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # `repo_id` is what the Hub upload targets, so it has to be the repo we
    # were asked for, not whatever local placeholder the converter wrote.
    ds = LeRobotDataset(repo_id, root=root)

    print(f"\nuploading to https://huggingface.co/datasets/{repo_id} ...")
    ds.push_to_hub(
        branch=branch,
        tags=tags,
        license=license_,
        private=private,
        push_videos=push_videos,
        upload_large_folder=large,
    )
    print(f"done: https://huggingface.co/datasets/{repo_id}")


def verify(repo_id: str) -> None:
    """Confirm the files that carry the labels actually landed on the Hub."""
    from huggingface_hub import HfApi

    files = set(HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset"))
    for name in ("meta/info.json", "meta/tasks.parquet", "meta/subtasks.parquet"):
        print(f"  {'ok  ' if name in files else 'MISSING'}  {name}")
    episodes = [f for f in files if f.startswith("meta/episodes/")]
    print(f"  {'ok  ' if episodes else 'MISSING'}  meta/episodes/ ({len(episodes)} files)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True,
                    help="converted dataset directory")
    ap.add_argument("--repo-id", required=True, help="Hub repo, e.g. user/xarm7-kitting")
    ap.add_argument("--dry-run", action="store_true",
                    help="inspect and check, upload nothing")
    ap.add_argument("--public", action="store_true",
                    help="publish publicly (default is private)")
    ap.add_argument("--branch", help="push to this branch instead of main")
    ap.add_argument("--tags", nargs="*", default=["robotics", "xarm7", "manipulation"])
    ap.add_argument("--license", default="apache-2.0")
    ap.add_argument("--no-videos", action="store_true",
                    help="upload metadata and parquet only, skip videos/")
    ap.add_argument("--no-large-folder", action="store_true",
                    help="use upload_folder instead of the resumable uploader")
    ap.add_argument("--force", action="store_true",
                    help="upload even though the subtask checks failed")
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"no such directory: {root}")

    summary = dataset_summary(root)
    print_summary(root, args.repo_id, summary, not args.public)

    problems = check_subtasks(summary)
    if problems:
        print("\nsubtask check FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        print("\nThis looks like a conversion that dropped the editor's labels."
              "\nRe-run convert_to_lerobot.py, or pass --force to upload anyway.")
        if not args.force:
            raise SystemExit(1)
    else:
        print("\nsubtask check ok")

    if args.dry_run:
        print("\ndry run: nothing uploaded")
        return

    if args.public:
        # Public is irreversible in practice -- anyone can have pulled it by
        # the time you change your mind. Make it a deliberate keystroke.
        answer = input(f"\npublish {args.repo_id} PUBLICLY? type 'public' to confirm: ")
        if answer.strip() != "public":
            raise SystemExit("aborted")

    upload(root, args.repo_id, not args.public, args.tags, args.license,
           args.branch, not args.no_videos, not args.no_large_folder)

    print("\nverifying uploaded metadata:")
    verify(args.repo_id)


if __name__ == "__main__":
    sys.exit(main())
