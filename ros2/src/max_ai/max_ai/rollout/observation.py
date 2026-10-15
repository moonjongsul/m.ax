"""원시 센서값 -> 정책 입력. max_data_collect의 frame_builder와 같은 변환.

정책은 녹화된 데이터로 학습했으므로, 여기서의 변환은 녹화 파이프라인
(`collector/frame_builder.py` + `convert_to_lerobot.py`)과 한 치도 다르면
안 된다. 대응 관계:

    observation.state (18)  [x y z]             <- /xarm/eef_pose position (m, xarm_base)
                            [r11 r21 r31
                             r12 r22 r32]       <- 같은 pose의 쿼터니언 -> rot6d
                            j1..j7              <- /xarm/joint_states position (rad)
                                                   (ROS 이름은 joint1..joint7)
                            gripper             <- gripper joint position, 0..1 그대로
                            gripper_effort      <- clamp(effort, 0, 600) / 600
    observation.images.*    CompressedImage -> BGR 디코드 -> 회전 -> 640x480 INTER_AREA
                            -> RGB, CHW float [0, 1]

이미지 키는 데이터셋 이름(env, wrist_front, wrist_rear) 그대로 넘긴다.
camera1/2/3으로의 이름 바꾸기는 체크포인트에 저장된 전처리기가 한다 --
여기서 또 바꾸면 두 번 바뀐다.

ROS를 import하지 않는다. 노드는 메시지에서 숫자만 꺼내 `RawObservation`을
채우고, replay는 데이터셋에서 같은 것을 채운다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

_ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}

STATE_KEY = "observation.state"
IMAGE_PREFIX = "observation.images."


@dataclass
class RawObservation:
    """한 시점의 센서값. 시각은 전부 같은 단조 시계(초) 기준."""

    stamp: float                                   # 가장 오래된 소스의 수신 시각
    images: dict[str, np.ndarray] = field(default_factory=dict)  # name -> HxWx3 uint8 BGR
    eef_twist: np.ndarray = field(default_factory=lambda: np.zeros(6))      # m/s, rad/s
    eef_position: np.ndarray = field(default_factory=lambda: np.zeros(3))   # m
    eef_quat: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    joint_position: np.ndarray = field(default_factory=lambda: np.zeros(7))  # rad, j1..j7 순서
    gripper_position: float = 0.0                  # 0 닫힘 .. 1 열림
    gripper_effort_raw: float = 0.0

    def effort_normalized(self, effort_max: float = 600.0) -> float:
        # 대기 중 effort는 약간 음수(-23..-16)로 읽힌다. 방향이 아니라 노이즈라
        # 0으로 자른다 -- frame_builder와 같은 처리.
        return float(min(1.0, max(0.0, self.gripper_effort_raw / effort_max)))


def prepare_image(bgr: np.ndarray, size: tuple[int, int] = (640, 480), rotate: int = 0) -> np.ndarray:
    """녹화 때와 같은 순서로 회전·리사이즈. 반환은 HxWx3 uint8 BGR."""
    if rotate in _ROTATE:
        bgr = cv2.rotate(bgr, _ROTATE[rotate])
    w, h = size
    if w > 0 and h > 0 and (bgr.shape[1], bgr.shape[0]) != (w, h):
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    return bgr


def decode_compressed(data: bytes) -> np.ndarray | None:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return img


STATE_DIM = 18


def quat_to_rot6d(quat) -> np.ndarray:
    """[qx qy qz qw] -> [r11 r21 r31 r12 r22 r32] (회전행렬의 1·2열).

    max_data_collect/collector/rotation.py의 같은 이름 함수와 같은 결과를
    내야 한다. 여기서 다시 구현하는 것은 이 패키지가 max_data_collect를
    import하지 않기 때문이고, 두 구현은 tests/test_core.py가 묶어서 비교한다.
    """
    qx, qy, qz, qw = (float(v) for v in np.asarray(quat).reshape(4))
    n = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        1.0 - 2.0 * (qy * qy + qz * qz),  # r11
        2.0 * (qx * qy + qz * qw),        # r21
        2.0 * (qx * qz - qy * qw),        # r31
        2.0 * (qx * qy - qz * qw),        # r12
        1.0 - 2.0 * (qx * qx + qz * qz),  # r22
        2.0 * (qy * qz + qx * qw),        # r32
    ], dtype=np.float32)


def state_vector(raw: RawObservation, effort_max: float = 600.0) -> np.ndarray:
    """데이터셋의 observation.state 18채널, 같은 순서.

    convert_to_lerobot_config.yaml의 `observation.state.main`과 짝이다.
    eef_twist는 들어가지 않는다 -- action(cartesian velocity)과 채널별
    상관이 0.84~0.99라서, 학습이 카메라를 읽는 대신 직전 명령을 되풀이하는
    지름길로 빠진다.
    """
    joint = np.asarray(raw.joint_position, dtype=np.float32).reshape(-1)
    if joint.shape[0] != 7:
        raise ValueError(f"joint_position must be 7 long, got {joint.shape[0]}")
    return np.concatenate([
        np.asarray(raw.eef_position, dtype=np.float32).reshape(3),
        quat_to_rot6d(raw.eef_quat),
        joint,
        np.array([raw.gripper_position, raw.effort_normalized(effort_max)], dtype=np.float32),
    ])


def to_policy_observation(
    raw: RawObservation,
    camera_names: list[str],
    image_size: tuple[int, int] = (640, 480),
    rotate: dict[str, int] | None = None,
    effort_max: float = 600.0,
) -> dict[str, torch.Tensor]:
    """배치 차원 없는 정책 입력. 전처리기(`to_batch_processor`)가 배치로 만든다."""
    rotate = rotate or {}
    obs: dict[str, torch.Tensor] = {STATE_KEY: torch.from_numpy(state_vector(raw, effort_max))}
    for name in camera_names:
        if name not in raw.images:
            raise KeyError(f"camera '{name}' missing from observation")
        bgr = prepare_image(raw.images[name], image_size, rotate.get(name, 0))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        obs[IMAGE_PREFIX + name] = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
    return obs
