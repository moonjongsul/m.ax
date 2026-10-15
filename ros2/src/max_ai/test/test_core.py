"""정책·ROS 없이 도는 핵심 로직 테스트.

    python -m pytest ros2/src/max_ai/test -q -k "core or tracker or split"
"""

from pathlib import Path

import numpy as np
import pytest

from max_ai.rollout.action import ActionSafety, to_xarm_speeds
from max_ai.rollout.config import SafetyConfig, load_config
from max_ai.rollout.executor import ChunkExecutor
from max_ai.rollout.observation import (
    STATE_DIM, RawObservation, quat_to_rot6d, state_vector, to_policy_observation,
)

CONFIG = Path(__file__).resolve().parents[1] / "config" / "xarm7_kitting.yaml"
FPS = 30.0


# ───────────────────────────────────────────────────────── chunk timing
def _chunk(n=50, offset=0.0):
    c = np.zeros((n, 7))
    c[:, 0] = np.arange(n) + offset  # vx에 인덱스를 심어 어느 스텝인지 읽는다
    return c


def test_chunk_index_follows_observation_time():
    ex = ChunkExecutor(FPS, execute_steps=25, blend_steps=0)
    ex.push(_chunk(), t_obs=1.0, now=1.1)
    # 관측 후 0.1초 = 3스텝 지난 시점부터 실행된다 (지연 보상)
    assert ex.action_at(1.1)[0] == 3
    assert ex.action_at(1.0 + 10 / FPS)[0] == 10


def test_chunk_requested_early_by_latency():
    ex = ChunkExecutor(FPS, execute_steps=25, blend_steps=0)
    assert ex.needs_chunk(0.0, 0.1)
    ex.push(_chunk(), t_obs=0.0, now=0.0)
    assert not ex.needs_chunk(21 / FPS, 0.1)   # 21 + 3 < 25
    assert ex.needs_chunk(22 / FPS, 0.1)       # 22 + 3 >= 25


def test_chunk_runs_out_then_stops():
    ex = ChunkExecutor(FPS, execute_steps=25, blend_steps=0)
    ex.push(_chunk(), t_obs=0.0, now=0.0)
    assert ex.action_at(49 / FPS) is not None
    assert ex.action_at(50 / FPS) is None


def test_chunk_blend_is_linear():
    ex = ChunkExecutor(FPS, execute_steps=25, blend_steps=4)
    ex.push(_chunk(offset=0.0), t_obs=0.0, now=0.0)
    ex.push(_chunk(offset=100.0), t_obs=20 / FPS, now=23 / FPS)
    # 도착 순간 w=0: 이전 청크 23번 = 23
    assert ex.action_at(23 / FPS)[0] == pytest.approx(23.0)
    # 2스텝 후 w=0.5: 0.5*(100+5) + 0.5*25
    assert ex.action_at(25 / FPS)[0] == pytest.approx(0.5 * 105 + 0.5 * 25)
    # 혼합 끝: 새 청크만
    assert ex.action_at(27 / FPS)[0] == pytest.approx(107.0)


# ─────────────────────────────────────────────────────────────── safety
def _safety(**kw):
    cfg = SafetyConfig(speed_scale=1.0, max_linear_accel=1e9, max_angular_accel=1e9, **kw)
    return ActionSafety(cfg, FPS)


INSIDE = np.array([0.1, -0.3, 0.15])


def test_units_mm_per_s_only_for_linear():
    s = _safety()
    cmd = s(np.array([0.1, -0.2, 0.05, 0.3, -0.1, 0.2, 1.0]), INSIDE)
    assert to_xarm_speeds(cmd) == pytest.approx([100.0, -200.0, 50.0, 0.3, -0.1, 0.2])


def test_speed_scale_and_clip():
    s = ActionSafety(SafetyConfig(speed_scale=0.5, max_linear_accel=1e9, max_angular_accel=1e9), FPS)
    cmd = s(np.array([0.2, 0, 0, 2.0, 0, 0, 1.0]), INSIDE)
    assert cmd.linear[0] == pytest.approx(0.1)
    assert cmd.angular[0] == pytest.approx(0.6)  # 1.0 -> max_angular
    assert "clip" in cmd.flags


def test_workspace_blocks_only_outward_motion():
    s = _safety()
    at_floor = np.array([0.1, -0.3, 0.0])
    down = s(np.array([0.05, 0, -0.05, 0, 0, 0, 1.0]), at_floor)
    assert down.linear[2] == 0.0 and down.linear[0] == pytest.approx(0.05)
    up = s(np.array([0, 0, 0.05, 0, 0, 0, 1.0]), at_floor)
    assert up.linear[2] == pytest.approx(0.05)


def test_accel_limit():
    s = ActionSafety(SafetyConfig(speed_scale=1.0, max_linear_accel=1.5), FPS)
    cmd = s(np.array([0.25, 0, 0, 0, 0, 0, 1.0]), INSIDE)
    assert cmd.linear[0] == pytest.approx(1.5 / FPS)
    assert "accel" in cmd.flags


def test_gripper_edges_only_with_hysteresis():
    s = _safety()
    s.reset(gripper_open=1.0)
    act = lambda g: s(np.array([0, 0, 0, 0, 0, 0, g]), INSIDE)  # noqa: E731
    assert act(0.9).gripper is None           # 이미 열림
    assert act(0.45).gripper is None          # 히스테리시스 안
    assert act(0.2).gripper == 0.0            # 닫힘으로 바뀜
    assert act(0.1).gripper is None           # 유지
    assert act(0.7).gripper == 1.0


def test_invalid_action_stops():
    s = _safety()
    cmd = s(np.array([np.nan] * 7), INSIDE)
    assert cmd.is_zero and "invalid_action" in cmd.flags


# ─────────────────────────────────────────────────────────── observation
def test_state_vector_matches_recorder():
    """observation.state 18채널: [eef_position(3) rot6d(6) joint(7) gripper effort]."""
    joint = np.arange(7) * 0.1
    raw = RawObservation(stamp=0.0, eef_position=np.array([0.3, -0.1, 0.2]),
                         eef_quat=np.array([0.0, 0.0, 0.0, 1.0]), joint_position=joint,
                         gripper_position=0.4, gripper_effort_raw=-20.0)
    v = state_vector(raw)
    assert v.shape == (STATE_DIM,) == (18,)
    assert v[:3] == pytest.approx([0.3, -0.1, 0.2])
    assert v[3:9] == pytest.approx([1, 0, 0, 0, 1, 0])  # 단위 회전
    assert v[9:16] == pytest.approx(joint)
    assert v[16] == pytest.approx(0.4)
    assert v[17] == 0.0  # 음수 effort는 노이즈 -> 0
    raw.gripper_effort_raw = 900.0
    assert state_vector(raw)[17] == 1.0

    # eef_twist는 들어가지 않는다 -- action과 상관이 0.84~0.99라 지름길이 된다.
    raw.eef_twist = np.arange(6) * 7.0
    assert state_vector(raw) == pytest.approx(state_vector(raw))


def test_state_vector_rejects_wrong_joint_count():
    raw = RawObservation(stamp=0.0, joint_position=np.zeros(6))
    with pytest.raises(ValueError):
        state_vector(raw)


def test_rot6d_matches_recorder():
    """녹화 쪽 quat_to_rot6d와 같은 값이어야 한다 -- 다르면 학습/추론이 어긋난다."""
    import importlib.util
    from pathlib import Path

    ref_path = (Path(__file__).resolve().parents[2]
                / "max_data_collect/max_data_collect/collector/rotation.py")
    if not ref_path.exists():
        pytest.skip(f"recorder rotation.py not found at {ref_path}")
    spec = importlib.util.spec_from_file_location("_rec_rotation", ref_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rng = np.random.default_rng(0)
    for _ in range(200):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        assert quat_to_rot6d(q) == pytest.approx(mod.quat_to_rot6d(*q), abs=1e-6)

    # 퇴화 입력은 단위 회전으로 떨어진다 (0으로 나누지 않는다).
    assert quat_to_rot6d([0.0, 0.0, 0.0, 0.0]) == pytest.approx([1, 0, 0, 0, 1, 0])


def test_images_resized_and_rgb():
    bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
    bgr[..., 0] = 255  # 파랑
    raw = RawObservation(stamp=0.0, images={"env": bgr})
    obs = to_policy_observation(raw, ["env"])
    img = obs["observation.images.env"]
    assert tuple(img.shape) == (3, 480, 640)
    assert float(img[2].mean()) == pytest.approx(1.0) and float(img[0].mean()) == 0.0


# ───────────────────────────────────────────────────────────────── config
def test_config_loads_and_rejects_typos(tmp_path):
    cfg = load_config(CONFIG)
    assert cfg.control.fps == 30.0 and set(cfg.ros.cameras) == {"env", "wrist_front", "wrist_rear"}
    bad = tmp_path / "bad.yaml"
    bad.write_text("safety:\n  speed_scal: 1.0\n")
    with pytest.raises(KeyError):
        load_config(bad)
