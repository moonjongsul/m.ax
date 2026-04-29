"""
Augment a joint-space LeRobot dataset with EEF task-space representations.

The original joint-space `action` / `observation.state` are renamed to
`action.joint_position` / `observation.state.joint_position`, and the
standard `action` / `observation.state` keys are overwritten with the
6D rotation EEF representation so that LeRobot policies (which read the
canonical `action` / `observation.state` keys) train on the rot6d
representation by default.

Resulting frame keys:

  - action                                : [x, y, z, r11..r23, gripper]      (10D)  ← rot6d, used by policy
  - action.joint_position                 : [j1..j7, gripper]                 (8D)
  - action.eef_quaternion                 : [x, y, z, qx, qy, qz, qw, gripper](8D)
  - action.eef_rotation_matrix            : [x, y, z, r11..r33, gripper]      (13D)
  - action.eef_rotation_6d                : [x, y, z, r11..r23, gripper]      (10D)
  - observation.state                          : [x, y, z, r11..r23, gripper] (10D)  ← rot6d, used by policy
  - observation.state.joint_position           : [j1..j7, gripper]            (8D)
  - observation.state.eef_quaternion           : [x, y, z, qx, qy, qz, qw, gripper] (8D)
  - observation.state.eef_rotation_matrix      : [x, y, z, r11..r33, gripper] (13D)
  - observation.state.eef_rotation_6d          : [x, y, z, r11..r23, gripper] (10D)

FK is computed via pinocchio using the FR3 URDF.
Gripper action is binarized: >= 0.8 -> 1.0, < 0.8 -> 0.0
Gripper state is kept as-is.
Rotation matrix is row-major flattened (numpy default): [r11, r12, r13, r21, r22, r23, r31, r32, r33].
Rotation 6D keeps the first two rows of the rotation matrix (Zhou et al. 2019).

Usage:
    python scripts/data_conversion.py --repo-id moonjongsul/manufacturing_kitting_dataset
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pinocchio as pin
from huggingface_hub import HfApi, snapshot_download
from scipy.spatial.transform import Rotation
from tqdm import tqdm

URDF_PATH = Path(__file__).parent / "fr3.urdf"
TCP_FRAME = "fr3_link8"
GRIPPER_THRESHOLD = 0.8

OUTPUT_BRANCH = "eef_augmented"

JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6", "j7", "gripper_width"]
QUAT_NAMES = ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper_width"]
ROTMAT_NAMES = [
    "x", "y", "z",
    "r11", "r12", "r13",
    "r21", "r22", "r23",
    "r31", "r32", "r33",
    "gripper_width",
]
ROT6D_NAMES = [
    "x", "y", "z",
    "r11", "r12", "r13",
    "r21", "r22", "r23",
    "gripper_width",
]


class FR3Kinematics:
    """Forward kinematics for Franka FR3 using pinocchio."""

    def __init__(self, urdf_path: str | Path = URDF_PATH):
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.frame_id = self.model.getFrameId(TCP_FRAME)

    def forward(self, joint_positions_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute FK for 7-DOF joint positions in radians.

        Returns:
            position: (3,) xyz in base frame
            rotation: (3, 3) rotation matrix
        """
        q = pin.neutral(self.model)
        q[:7] = joint_positions_rad[:7]

        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        pose = self.data.oMf[self.frame_id]
        return pose.translation.copy(), pose.rotation.copy()


def binarize_gripper(gripper_value: float) -> float:
    """Binarize gripper action: >= 0.8 -> 1.0, < 0.8 -> 0.0"""
    return 1.0 if gripper_value >= GRIPPER_THRESHOLD else 0.0


def joint_to_eef_representations(
    joint_positions_rad: np.ndarray,
    gripper: float,
    fk: FR3Kinematics,
    is_action: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute joint / eef_quaternion / eef_rotation_matrix / eef_rotation_6d vectors from one frame.

    Returns:
        joint     : (8,)  [j1..j7, gripper]
        eef_quat  : (8,)  [x, y, z, qx, qy, qz, qw, gripper]
        eef_rotmat: (13,) [x, y, z, r11..r33 (row-major), gripper]
        eef_rot6d : (10,) [x, y, z, r11..r23 (first two rows, row-major), gripper]
    """
    pos, rot = fk.forward(joint_positions_rad)

    g = binarize_gripper(gripper) if is_action else gripper

    joint_vec = np.concatenate([joint_positions_rad[:7], [g]]).astype(np.float32)

    quat = Rotation.from_matrix(rot).as_quat()  # (x, y, z, w)
    eef_quat = np.concatenate([pos, quat, [g]]).astype(np.float32)

    eef_rotmat = np.concatenate([pos, rot.flatten(), [g]]).astype(np.float32)

    eef_rot6d = np.concatenate([pos, rot[:2, :].flatten(), [g]]).astype(np.float32)

    return joint_vec, eef_quat, eef_rotmat, eef_rot6d


def convert_parquet(pq_path: Path, fk: FR3Kinematics) -> int:
    """Augment a single parquet file in-place with EEF representations.

    Returns number of frames processed.
    """
    df = pd.read_parquet(pq_path)

    states = np.array(df["observation.state"].tolist())
    actions = np.array(df["action"].tolist())

    state_joint, state_quat, state_rotmat, state_rot6d = [], [], [], []
    action_joint, action_quat, action_rotmat, action_rot6d = [], [], [], []

    for i in range(len(df)):
        sj, sq, sr, s6 = joint_to_eef_representations(
            states[i, :7], float(states[i, 7]), fk, is_action=False
        )
        aj, aq, ar, a6 = joint_to_eef_representations(
            actions[i, :7], float(actions[i, 7]), fk, is_action=True
        )
        state_joint.append(sj.tolist())
        state_quat.append(sq.tolist())
        state_rotmat.append(sr.tolist())
        state_rot6d.append(s6.tolist())
        action_joint.append(aj.tolist())
        action_quat.append(aq.tolist())
        action_rotmat.append(ar.tolist())
        action_rot6d.append(a6.tolist())

    # Joint data is preserved under explicit `*.joint_position` keys.
    df["observation.state.joint_position"] = state_joint
    df["observation.state.eef_quaternion"] = state_quat
    df["observation.state.eef_rotation_matrix"] = state_rotmat
    df["observation.state.eef_rotation_6d"] = state_rot6d
    df["action.joint_position"] = action_joint
    df["action.eef_quaternion"] = action_quat
    df["action.eef_rotation_matrix"] = action_rotmat
    df["action.eef_rotation_6d"] = action_rot6d
    # Canonical `action` / `observation.state` keys are overwritten with rot6d,
    # so LeRobot policies pick up the 6D rotation representation by default.
    df["observation.state"] = state_rot6d
    df["action"] = action_rot6d

    df.to_parquet(pq_path, index=False)

    return len(df)


def _compute_stats(arr: np.ndarray) -> dict:
    return {
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
    }


def convert_dataset(
    repo_id: str | None = None,
    output_dir: str | None = None,
    input_dir: str | None = None,
    push: bool = True,
):
    """
    Augment a LeRobot dataset with EEF keys (quaternion + rotation matrix).

    Source can be either:
      - a HuggingFace repo (`repo_id`), or
      - a local directory (`input_dir`).

    Output is written to `output_dir` (when given), otherwise a /tmp working dir.
    Pushes to HuggingFace branch `eef_augmented` only when `push=True` and
    `repo_id` is provided.
    """
    if input_dir is not None:
        # Local mode
        src_dir = Path(input_dir)
        if not src_dir.exists():
            raise FileNotFoundError(f"Input dir not found: {src_dir}")
        print(f"[1/4] Local source dataset: {src_dir}")

        if output_dir is None:
            raise ValueError("--output-dir is required when --input-dir is set")
        dst_dir = Path(output_dir)
    else:
        # HuggingFace mode
        if repo_id is None:
            raise ValueError("Either --repo-id or --input-dir must be provided")
        print(f"[1/4] Downloading source dataset: {repo_id} (main branch)")
        src_dir = Path(snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision="main",
        ))
        print(f"  Source: {src_dir}")

        if output_dir is None:
            dst_dir = Path(f"/tmp/lerobot_conversion/{repo_id.replace('/', '_')}_{OUTPUT_BRANCH}") / "dataset"
        else:
            dst_dir = Path(output_dir)

    # Step 2: Copy to working directory
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir, ignore=shutil.ignore_patterns(".git*"))
    print(f"[2/4] Working copy: {dst_dir}")

    # Step 3: Augment parquet files
    print("[3/4] Augmenting parquet data with EEF representations...")
    fk = FR3Kinematics()

    parquet_files = sorted(dst_dir.glob("data/**/*.parquet"))
    total_frames = 0
    for pq_path in tqdm(parquet_files, desc="Converting"):
        total_frames += convert_parquet(pq_path, fk)
    print(f"  Augmented {total_frames} frames across {len(parquet_files)} files")

    # Update info.json: canonical action/observation.state become rot6d (10D),
    # joint data is preserved under *.joint_position keys.
    info_path = dst_dir / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    joint_template = {
        "dtype": "float32",
        "shape": [8],
        "names": JOINT_NAMES,
    }
    base_template = {
        "dtype": "float32",
        "shape": [8],
        "names": QUAT_NAMES,
    }
    rotmat_template = {
        "dtype": "float32",
        "shape": [13],
        "names": ROTMAT_NAMES,
    }
    rot6d_template = {
        "dtype": "float32",
        "shape": [10],
        "names": ROT6D_NAMES,
    }

    # Canonical keys (read by LeRobot policies) are now rot6d.
    info["features"]["observation.state"] = dict(rot6d_template)
    info["features"]["action"] = dict(rot6d_template)
    # Preserve original joint data under explicit keys.
    info["features"]["observation.state.joint_position"] = dict(joint_template)
    info["features"]["action.joint_position"] = dict(joint_template)
    info["features"]["observation.state.eef_quaternion"] = dict(base_template)
    info["features"]["action.eef_quaternion"] = dict(base_template)
    info["features"]["observation.state.eef_rotation_matrix"] = dict(rotmat_template)
    info["features"]["action.eef_rotation_matrix"] = dict(rotmat_template)
    info["features"]["observation.state.eef_rotation_6d"] = dict(rot6d_template)
    info["features"]["action.eef_rotation_6d"] = dict(rot6d_template)

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print("  Updated info.json")

    # Recompute stats.json for every state/action variant.
    print("  Recomputing stats.json...")
    buckets = {
        "observation.state": [],
        "observation.state.joint_position": [],
        "observation.state.eef_quaternion": [],
        "observation.state.eef_rotation_matrix": [],
        "observation.state.eef_rotation_6d": [],
        "action": [],
        "action.joint_position": [],
        "action.eef_quaternion": [],
        "action.eef_rotation_matrix": [],
        "action.eef_rotation_6d": [],
    }
    for pq_path in parquet_files:
        df = pd.read_parquet(pq_path)
        for k in buckets:
            buckets[k].extend(df[k].tolist())

    stats_path = dst_dir / "meta" / "stats.json"
    with open(stats_path) as f:
        stats = json.load(f)

    for k, vals in buckets.items():
        arr = np.array(vals, dtype=np.float32)
        stats[k] = _compute_stats(arr)

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print("  Updated stats.json")

    # Step 4: Push to HuggingFace (optional)
    if push and repo_id is not None:
        print(f"[4/4] Pushing to HuggingFace branch: {OUTPUT_BRANCH}")
        api = HfApi()

        try:
            api.create_branch(repo_id=repo_id, branch=OUTPUT_BRANCH, repo_type="dataset")
            print(f"  Created branch: {OUTPUT_BRANCH}")
        except Exception:
            print(f"  Branch '{OUTPUT_BRANCH}' already exists")

        api.upload_folder(
            folder_path=str(dst_dir),
            repo_id=repo_id,
            repo_type="dataset",
            revision=OUTPUT_BRANCH,
            commit_message="Augment dataset with EEF quaternion / rotation-matrix representations via FK (pinocchio)",
        )
        print(f"\nDone! Dataset: {repo_id} (branch: {OUTPUT_BRANCH})")
    else:
        print(f"[4/4] Skipped HF push. Local output: {dst_dir}")
        print(f"\nDone! Local dataset: {dst_dir}")

    print(f"  observation.state                      [10D]:  {ROT6D_NAMES}  (canonical, rot6d)")
    print(f"  observation.state.joint_position        [8D]:  {JOINT_NAMES}")
    print(f"  observation.state.eef_quaternion        [8D]:  {QUAT_NAMES}")
    print(f"  observation.state.eef_rotation_matrix  [13D]:  {ROTMAT_NAMES}")
    print(f"  observation.state.eef_rotation_6d      [10D]:  {ROT6D_NAMES}")
    print(f"  action                                 [10D]:  {ROT6D_NAMES}  (canonical, rot6d)")
    print(f"  action.joint_position                   [8D]:  {JOINT_NAMES}")
    print(f"  action.eef_quaternion                   [8D]:  {QUAT_NAMES}")
    print(f"  action.eef_rotation_matrix             [13D]:  {ROTMAT_NAMES}")
    print(f"  action.eef_rotation_6d                 [10D]:  {ROT6D_NAMES}")


def main():
    parser = argparse.ArgumentParser(
        description="Augment joint-space dataset with EEF quaternion + rotation-matrix keys"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="HuggingFace dataset repo ID (omit when using --input-dir)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/workspace/m.ax/datasets/manufacturing_kitting_dataset",
        help="Local source dataset directory (skips HF download)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/m.ax/datasets/manufacturing_kitting_dataset_",
        help="Local output directory (required with --input-dir)",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Skip pushing to HuggingFace (always skipped when --input-dir is set without --repo-id)",
    )
    args = parser.parse_args()

    if args.repo_id is None and args.input_dir is None:
        parser.error("either --repo-id or --input-dir must be provided")

    convert_dataset(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        input_dir=args.input_dir,
        push=not args.no_push,
    )


if __name__ == "__main__":
    main()
