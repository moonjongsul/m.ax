"""Turn one Communicator snapshot into one recorded row.

Shapes follow the agreed schema exactly; a stream that has not published
records zeros rather than shortening the row, because every array in the
HDF5 must stay length N. The `stale` bitmask is what says a value was
held over rather than freshly measured.
"""

import cv2
import numpy as np

from max_data_collect.collector.communicator import (
    ACTION, CAM_PREFIX, EEF_POSE, EEF_TWIST, GRIPPER, JOINT,
)
from max_data_collect.collector.rotation import quat_to_rot6d


_ROTATE_CODES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class FrameBuilder:
    """Builds rows; holds only the per-episode frame counter."""

    def __init__(self, logger, cfg: dict):
        self._log = logger
        self._cameras = cfg["observation_cameras"]
        self._cam_names = list(self._cameras)

        manip = cfg["observation_manipulator"]
        self._joint_names = list(manip["joint"]["joint_names"])
        self._record_joint_vel = bool(manip["joint"].get("record_velocity", True))
        self._pose_frame = manip["eef_position"].get("expected_frame_id", "")
        self._twist_frame = manip["eef_velocity"].get("expected_frame_id", "")

        grip = cfg["observation_gripper"]
        self._grip_joint = grip["joint_name"]
        self._grip_pos_field = grip.get("position", {}).get("field", "position")
        effort = grip["effort"]
        self._grip_eff_field = effort.get("field", "effort")
        self._eff_min = float(effort["effort_min"])
        self._eff_max = float(effort["effort_max"])
        self._eff_clip = bool(effort.get("clip", True))

        self._staleness = float(cfg["recording"]["staleness_timeout"])

        # Bit per stream in the `stale` mask, stable across episodes so a
        # recorded mask stays readable without consulting the config.
        self._bits = {}
        for i, name in enumerate(self._cam_names):
            self._bits[CAM_PREFIX + name] = i
        base = len(self._cam_names)
        for i, key in enumerate((JOINT, EEF_POSE, EEF_TWIST, GRIPPER, ACTION)):
            self._bits[key] = base + i

        self._warned: set[str] = set()
        self.reset()

    def reset(self) -> None:
        self._index = 0

    @property
    def stale_bits(self) -> dict:
        return dict(self._bits)

    # ── build ──────────────────────────────────────────────────────────

    def build(self, snapshot: dict, now: float, episode_start: float) -> dict:
        stale_mask = 0
        for key, (msg, _stamp, recv) in snapshot.items():
            if msg is None or (now - recv) > self._staleness:
                stale_mask |= 1 << self._bits[key]

        joint = snapshot[JOINT][0]
        pose = snapshot[EEF_POSE][0]
        twist = snapshot[EEF_TWIST][0]
        grip = snapshot[GRIPPER][0]
        action = snapshot[ACTION][0]

        # The action stream is the reference axis for the dataset, so its
        # stamp is the row's stamp when it is present.
        stamp_ns = snapshot[ACTION][1] or snapshot[EEF_POSE][1] or 0

        row = {
            "index": self._index,
            "timestamp": now - episode_start,
            "stamp_ns": stamp_ns,
            "stale": stale_mask,
            "images": {
                name: self._image(name, snapshot[CAM_PREFIX + name][0])
                for name in self._cam_names
            },
            "observation.joint_position": self._joint_field(joint, "position"),
            "observation.eef_position": self._eef_position(pose),
            "observation.eef_orientation": self._eef_orientation(pose),
            "observation.eef_rot6d": self._eef_rot6d(pose),
            "observation.eef_velocity": self._eef_velocity(twist),
            "observation.gripper_position": self._gripper_position(grip),
            "observation.gripper_effort": 0.0,
            "observation.gripper_effort_raw": self._gripper_effort_raw(grip),
            "action.cartesian_velocity": self._action_velocity(action),
            "action.gripper": self._action_gripper(action),
        }
        row["observation.gripper_effort"] = self._normalise_effort(
            row["observation.gripper_effort_raw"]
        )
        if self._record_joint_vel:
            row["observation.joint_velocity"] = self._joint_field(joint, "velocity")

        self._index += 1
        return row

    # ── images ─────────────────────────────────────────────────────────

    def _image(self, name: str, msg):
        """Decode to BGR and apply rotate/resize; None if nothing arrived.

        Decoding here (not in the writer) is deliberate: the video
        encoder wants raw frames anyway, and doing it once on the
        sampling thread avoids a second JPEG round-trip.
        """
        if msg is None:
            return None
        img = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            self._warn(f"camera '{name}': undecodable frame")
            return None

        cam = self._cameras[name]
        rotate = cam.get("rotate", 0)
        if rotate in _ROTATE_CODES:
            img = cv2.rotate(img, _ROTATE_CODES[rotate])
        width, height = cam.get("resolution", [0, 0])
        if width > 0 and height > 0 and (img.shape[1], img.shape[0]) != (width, height):
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        return img

    # ── manipulator ────────────────────────────────────────────────────

    def _joint_field(self, msg, field: str) -> list:
        """`joint_names` in configured order; missing joints record 0.0."""
        n = len(self._joint_names)
        if msg is None:
            return [0.0] * n
        values = getattr(msg, field, None) or []
        by_name = dict(zip(msg.name, values))
        missing = [j for j in self._joint_names if j not in by_name]
        if missing:
            self._warn(f"joint_states is missing {missing}")
        return [float(by_name.get(j, 0.0)) for j in self._joint_names]

    def _eef_position(self, msg) -> list:
        if msg is None:
            return [0.0, 0.0, 0.0]
        self._check_frame(msg, self._pose_frame, "eef_pose")
        p = msg.pose.position
        return [p.x, p.y, p.z]

    def _eef_orientation(self, msg) -> list:
        # Stored exactly as published; rot6d/euler is a training concern.
        if msg is None:
            return [0.0, 0.0, 0.0, 1.0]
        o = msg.pose.orientation
        return [o.x, o.y, o.z, o.w]

    def _eef_rot6d(self, msg) -> list:
        """The same rotation as `eef_orientation`, in 6D form.

        Redundant with the quaternion by construction -- kept because it
        is the form policies train on, and deriving it here means every
        consumer sees the identical values.
        """
        if msg is None:
            return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        o = msg.pose.orientation
        return quat_to_rot6d(o.x, o.y, o.z, o.w)

    def _eef_velocity(self, msg) -> list:
        if msg is None:
            return [0.0] * 6
        self._check_frame(msg, self._twist_frame, "eef_twist")
        lin, ang = msg.twist.linear, msg.twist.angular
        return [lin.x, lin.y, lin.z, ang.x, ang.y, ang.z]

    def _check_frame(self, msg, expected: str, label: str) -> None:
        actual = msg.header.frame_id
        if expected and actual and actual != expected:
            self._warn(
                f"{label} frame_id '{actual}' != expected '{expected}'"
            )

    # ── gripper ────────────────────────────────────────────────────────

    def _grip_value(self, msg, field: str) -> float:
        if msg is None:
            return 0.0
        values = getattr(msg, field, None) or []
        try:
            i = list(msg.name).index(self._grip_joint)
        except ValueError:
            self._warn(f"gripper joint '{self._grip_joint}' not in message")
            return 0.0
        return float(values[i]) if i < len(values) else 0.0

    def _gripper_position(self, msg) -> float:
        # Continuous 0..1 as published -- the commanded 1/0 lives on the
        # action side, and the two differ whenever a part is held.
        return self._grip_value(msg, self._grip_pos_field)

    def _gripper_effort_raw(self, msg) -> float:
        return self._grip_value(msg, self._grip_eff_field)

    def _normalise_effort(self, raw: float) -> float:
        # Idle effort reads slightly negative (sensor noise around zero),
        # so clipping to [min, max] lands it on 0.0; ~600 is a firm grasp.
        value = (raw - self._eff_min) / (self._eff_max - self._eff_min)
        return min(1.0, max(0.0, value)) if self._eff_clip else value

    # ── action ─────────────────────────────────────────────────────────

    def _action_velocity(self, msg) -> list:
        # m/s and rad/s, matching eef_twist -- no per-axis rescaling.
        if msg is None:
            return [0.0] * 6
        return [float(v) for v in msg.cartesian_velocity]

    def _action_gripper(self, msg) -> float:
        return 0.0 if msg is None else float(msg.gripper_width_percent)

    # ── logging ────────────────────────────────────────────────────────

    def _warn(self, message: str) -> None:
        """Warn once per distinct message; these repeat at 30 Hz."""
        if message not in self._warned:
            self._warned.add(message)
            self._log.warning(f"[collect] {message}")
