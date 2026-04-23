"""ROS2 토픽 구독 → LeRobot 포맷 에피소드 기록 도구.

설정은 `config/proj_gt_kitting.yaml`의 `data_collect:` 블록과 `env:` 블록에서 읽는다.
- action   : env.robot.topic.publish, env.gripper.topic.publish (gello 등 외부 노드가 발행 중)
- state    : env.robot.topic.subscribe (/joint_states), env.gripper.topic.subscribe
- images   : env.camera.*.topic.subscribe (CompressedImage)

키 입력:
- s      : 녹화 시작 (warmup 재시작, episode buffer clear)
- space  : 현재 에피소드 종료 → save_episode
- r      : 현재 에피소드 버리기 (clear_episode_buffer)
- q      : 완전 종료 (녹화 중이면 discard, dataset.finalize)
"""

from __future__ import annotations

import argparse
import os
import queue
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

# from common.utils.ros_utils import resolve_msg_type
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState, CompressedImage
from std_msgs.msg import Float32

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "thirdparty" / "lerobot" / "src"))
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402


# ─── 설정 파싱 ────────────────────────────────────────────────────────────────
MSG_TYPE_MAP = {
    'sensor_msgs/JointState':           JointState,
    'sensor_msgs/msg/JointState':       JointState,
    'std_msgs/Float32':                 Float32,
    'std_msgs/msg/Float32':             Float32,
    'sensor_msgs/CompressedImage':      CompressedImage,
    'sensor_msgs/msg/CompressedImage':  CompressedImage,
    'geometry_msgs/PoseStamped':        PoseStamped,
    'geometry_msgs/msg/PoseStamped':    PoseStamped,
}

@dataclass
class CameraSpec:
    name: str
    topic: str
    msg_type: str
    domain_id: int
    size: tuple[int, int]  # (W, H) 원본
    rotate: int


@dataclass
class ScalarTopicSpec:
    topic: str
    msg_type: str
    domain_id: int


@dataclass
class CollectConfig:
    hz: int
    task_pools: dict[str, list[str]]   # task_id → paraphrase pool (순서 보존)
    warmup_sec: float
    img_shape: tuple[int, int]  # (W, H) 저장·시각화용
    output_dir: Path
    repo_id: str

    robot_action: ScalarTopicSpec      # /gello/joint_states
    gripper_action: ScalarTopicSpec    # /gripper/.../target_gripper_width_percent
    robot_state: ScalarTopicSpec       # /joint_states
    gripper_state: ScalarTopicSpec     # /dxl_parallel_gripper/joint_states
    cameras: list[CameraSpec] = field(default_factory=list)

    @property
    def task_ids(self) -> list[str]:
        return list(self.task_pools.keys())


def _parse_img_shape(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def _first_topic(d: dict) -> tuple[str, dict]:
    """dict의 첫 항목을 (key, value)로 반환."""
    for k, v in d.items():
        return k, v
    raise ValueError("topic dict가 비어있습니다.")

def resolve_msg_type(type_str: str):
    if type_str not in MSG_TYPE_MAP:
        raise ValueError(f"Unsupported message type: '{type_str}'. "
                         f"Available: {list(MSG_TYPE_MAP.keys())}")
    return MSG_TYPE_MAP[type_str]

def load_config(path: Path) -> CollectConfig:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    dc = cfg["data_collect"]
    env = cfg["env"]

    robot_topic = env["robot"]["topic"]
    gripper_topic = env["gripper"]["topic"]
    robot_domain = robot_topic["ros_domain_id"]
    gripper_domain = gripper_topic["ros_domain_id"]

    robot_action_topic, robot_action_spec = _first_topic(robot_topic["publish"])
    gripper_action_topic, gripper_action_spec = _first_topic(gripper_topic["publish"])
    robot_state_topic, robot_state_spec = _first_topic(robot_topic["subscribe"])
    gripper_state_topic, gripper_state_spec = _first_topic(gripper_topic["subscribe"])

    cameras: list[CameraSpec] = []
    for _cam_key, cam in env["camera"].items():
        cam_domain = cam["topic"]["ros_domain_id"]
        cam_sub_topic, cam_sub_spec = _first_topic(cam["topic"]["subscribe"])
        cameras.append(CameraSpec(
            name=cam["name"],
            topic=cam_sub_topic,
            msg_type=cam_sub_spec["type"],
            domain_id=cam_domain,
            size=tuple(cam["size"]),
            rotate=int(cam.get("rotate", 0)),
        ))

    raw_task_prompt = dc.get("task_prompt")
    if not isinstance(raw_task_prompt, dict) or not raw_task_prompt:
        raise ValueError("data_collect.task_prompt 는 {task_id: [paraphrase...]} dict 여야 합니다.")
    task_pools: dict[str, list[str]] = {}
    for task_id, raw_pool in raw_task_prompt.items():
        tid = str(task_id)
        pool = [str(p).strip() for p in (raw_pool or []) if str(p).strip()]
        if not pool:
            pool = [tid.replace("_", " ").replace("-", " ")]
        task_pools[tid] = pool

    return CollectConfig(
        hz=int(dc["hz"]),
        task_pools=task_pools,
        warmup_sec=float(dc["warmup_sec"]),
        img_shape=_parse_img_shape(dc["img_shape"]),
        output_dir=Path(dc["output_dir"]),
        repo_id=str(dc["repo_id"]),
        robot_action=ScalarTopicSpec(robot_action_topic, robot_action_spec["type"], robot_domain),
        gripper_action=ScalarTopicSpec(gripper_action_topic, gripper_action_spec["type"], gripper_domain),
        robot_state=ScalarTopicSpec(robot_state_topic, robot_state_spec["type"], robot_domain),
        gripper_state=ScalarTopicSpec(gripper_state_topic, gripper_state_spec["type"], gripper_domain),
        cameras=cameras,
    )


# ─── 토픽 버퍼 (thread-safe latest) ───────────────────────────────────────────

class LatestBuffer:
    """(msg, ts) 쌍을 토픽별로 덮어쓰기 저장."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, tuple[object, float]] = {}

    def set(self, key: str, value: object) -> None:
        with self._lock:
            self._data[key] = (value, time.time())

    def snapshot(self) -> dict[str, tuple[object, float]]:
        with self._lock:
            return dict(self._data)


_ROT_MAP = {
    90:  cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def _make_image_callback(buf: LatestBuffer, key: str, rotate: int,
                         target_wh: tuple[int, int]):
    tgt_w, tgt_h = target_wh

    def cb(msg):
        arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return
        rot = _ROT_MAP.get(rotate)
        if rot is not None:
            img = cv2.rotate(img, rot)
        h, w = img.shape[:2]
        if (w, h) != (tgt_w, tgt_h):
            img = cv2.resize(img, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)
        buf.set(key, img)

    return cb


# ─── ROS 도메인 브릿지 ───────────────────────────────────────────────────────

class DomainSubscriber:
    """단일 domain_id용 ROS Context/Node/Executor 래퍼."""

    def __init__(self, domain_id: int, node_name: str) -> None:
        from rclpy._rclpy_pybind11 import SignalHandlerOptions
        self.domain_id = domain_id
        self._ctx = Context()
        rclpy.init(
            context=self._ctx,
            domain_id=domain_id,
            signal_handler_options=SignalHandlerOptions.NO,
        )
        self.node = Node(node_name, context=self._ctx)
        self._executor: SingleThreadedExecutor | None = None
        self._thread: threading.Thread | None = None

    def subscribe(self, topic: str, msg_type_str: str, callback) -> None:
        msg_type = resolve_msg_type(msg_type_str)
        self.node.create_subscription(msg_type, topic, callback, 10)

    def start(self) -> None:
        self._executor = SingleThreadedExecutor(context=self._ctx)
        self._executor.add_node(self.node)
        self._thread = threading.Thread(
            target=self._executor.spin, daemon=True,
            name=f"data-collect-domain{self.domain_id}",
        )
        self._thread.start()

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
        self.node.destroy_node()
        rclpy.shutdown(context=self._ctx)


# ─── 시각화 ──────────────────────────────────────────────────────────────────

_STALE_THRESHOLD_SEC = 0.5
_WINDOW_NAME = "data_collect"


def _fmt_array(arr: np.ndarray, prec: int = 3) -> str:
    return "[" + ", ".join(f"{x:+.{prec}f}" for x in arr) + "]"


def _put_text(img: np.ndarray, text: str, org: tuple[int, int],
              color: tuple[int, int, int] = (255, 255, 255),
              scale: float = 0.5) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 2, cv2.LINE_AA)


def _build_grid(cam_images: dict[str, np.ndarray], cam_order: list[str],
                cell_wh: tuple[int, int]) -> np.ndarray:
    cw, ch = cell_wh
    tiles: list[np.ndarray] = []
    for name in cam_order:
        tile = cam_images.get(name)
        if tile is None:
            tile = np.zeros((ch, cw, 3), dtype=np.uint8)
            _put_text(tile, f"{name}: NO SIGNAL", (10, ch // 2), (0, 0, 255), 1.4)
        else:
            tile = tile.copy()
            _put_text(tile, name, (12, 36), (0, 255, 255), 1.2)
        tiles.append(tile)

    while len(tiles) < 4:
        tiles.append(np.zeros((ch, cw, 3), dtype=np.uint8))

    top = np.hstack(tiles[:2])
    bot = np.hstack(tiles[2:4])
    return np.vstack([top, bot])


def render(
    cfg: CollectConfig,
    cam_images: dict[str, np.ndarray],
    cam_ts: dict[str, float],
    scalar_ts: dict[str, float],
    state_vec: np.ndarray | None,
    action_vec: np.ndarray | None,
    status: str,
    elapsed: float,
    frame_count: int,
    saved_episodes: int,
    current_episode: int,
    active_task: str,
    task_index: int,
    task_total: int,
    pending_saves: int = 0,
) -> np.ndarray:
    cam_order = [c.name for c in cfg.cameras]
    grid = _build_grid(cam_images, cam_order, cfg.img_shape)

    # 그리드 상단 헤더: 활성 task + 에피소드 카운터
    gh, gw = grid.shape[:2]
    header_h = 80
    header = np.zeros((header_h, gw, 3), dtype=np.uint8)
    _put_text(header, f"TASK: {active_task} [{task_index + 1}/{task_total}]",
              (16, 44), (0, 255, 255), 1.6)
    ep_text = f"saved: {saved_episodes}   current: #{current_episode}"
    if pending_saves > 0:
        ep_text += f"   (saving {pending_saves})"
    (tw, _), _ = cv2.getTextSize(ep_text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
    color = (100, 180, 255) if pending_saves > 0 else (255, 220, 120)
    _put_text(header, ep_text, (gw - tw - 16, 44), color, 1.1)
    grid = np.vstack([header, grid])

    banner_h = 220
    gh, gw = grid.shape[:2]
    banner = np.zeros((banner_h, gw, 3), dtype=np.uint8)

    now = time.time()

    # 라인 1: 상태
    line1 = f"HZ {cfg.hz} | {status} | elapsed {elapsed:5.1f}s | frames {frame_count}"
    _put_text(banner, line1, (12, 36), (255, 255, 255), 1.1)

    # 라인 2: state
    if state_vec is not None:
        arm = _fmt_array(state_vec[:7])
        grip = state_vec[7]
        _put_text(banner, f"state  q={arm}  grip={grip:+.3f}",
                  (12, 78), (180, 255, 180), 0.9)

    # 라인 3: action
    if action_vec is not None:
        arm = _fmt_array(action_vec[:7])
        grip = action_vec[7]
        _put_text(banner, f"action q={arm}  grip={grip:+.3f}",
                  (12, 118), (180, 200, 255), 0.9)

    # 라인 4: 토픽 stale 상태
    topic_status: list[tuple[str, str, float]] = [
        ("state",     cfg.robot_state.topic,   scalar_ts.get(cfg.robot_state.topic, 0.0)),
        ("g-state",   cfg.gripper_state.topic, scalar_ts.get(cfg.gripper_state.topic, 0.0)),
        ("action",    cfg.robot_action.topic,  scalar_ts.get(cfg.robot_action.topic, 0.0)),
        ("g-action",  cfg.gripper_action.topic, scalar_ts.get(cfg.gripper_action.topic, 0.0)),
    ]
    for cam in cfg.cameras:
        topic_status.append((f"cam:{cam.name}", cam.topic, cam_ts.get(cam.name, 0.0)))

    all_ok = True
    x = 12
    y = 162
    col_w = 260
    row_h = 36
    for label, _tpc, ts in topic_status:
        age = now - ts if ts > 0 else float("inf")
        ok = age < _STALE_THRESHOLD_SEC
        if not ok:
            all_ok = False
        color = (120, 255, 120) if ok else (120, 120, 255)
        tag = f"{label}:OK" if ok else (f"{label}:STALE {age:3.1f}s" if ts > 0 else f"{label}:--")
        _put_text(banner, tag, (x, y), color, 0.8)
        x += col_w
        if x > gw - col_w:
            x = 12
            y += row_h
            if y > banner_h - 10:
                break

    canvas = np.vstack([grid, banner])

    # 전체 프레임 테두리:
    #   stale       → 빨강 블링크 (2Hz)
    #   REC 중      → 녹색 블링크 (2Hz)
    #   그 외 OK    → 녹색 상시
    base_thickness = 12
    if not all_ok:
        border_color = (0, 0, 255)
        border_thickness = base_thickness
        draw = int(now * 4) % 2 == 0
    elif status == "REC":
        border_color = (0, 140, 255)  # 주황
        border_thickness = base_thickness * 4
        draw = int(now * 4) % 2 == 0
    else:
        border_color = (0, 220, 0)
        border_thickness = base_thickness
        draw = True
    if draw:
        h, w = canvas.shape[:2]
        cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), border_color, border_thickness)

    return canvas


# ─── 에피소드 레코더 (LeRobot wrapper) ────────────────────────────────────────

def build_features(cam_names: list[str], img_wh: tuple[int, int]) -> dict:
    w, h = img_wh
    features = {
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["joint1", "joint2", "joint3", "joint4",
                      "joint5", "joint6", "joint7", "gripper_width"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["joint1", "joint2", "joint3", "joint4",
                      "joint5", "joint6", "joint7", "gripper_width"],
        },
    }
    for name in cam_names:
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        }
    return features


class EpisodeRecorder:
    """LeRobotDataset 접근을 단일 워커 스레드로 직렬화.

    메인 루프는 `add(...)`, `save()`, `discard()`, `finalize()` 호출 시
    모두 커맨드만 큐에 넣고 즉시 return. 무거운 I/O(비디오 인코딩 등)는
    워커 스레드에서 순차 처리됨.
    """

    _CMD_SAVE = "save"
    _CMD_DISCARD = "discard"
    _CMD_FINALIZE = "finalize"

    def __init__(self, cfg: CollectConfig) -> None:
        self.cfg = cfg
        self._rng = random.Random()
        # 활성 task (←/→로 전환). 초기값은 dict 첫 key.
        self._active_task: str = next(iter(cfg.task_pools))
        cam_names = [c.name for c in cfg.cameras]
        features = build_features(cam_names, cfg.img_shape)

        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        dataset_root = cfg.output_dir / cfg.repo_id
        self.dataset_root = dataset_root

        # 로컬 데이터셋 유효성 판정: meta/info.json 과 meta/tasks.parquet 이 모두 존재해야 완전한 상태.
        # (에피소드 저장 도중 중단된 경우 info.json만 남을 수 있음 → 원격 fallback 방지 위해 정리)
        info_file = dataset_root / "meta" / "info.json"
        tasks_file = dataset_root / "meta" / "tasks.parquet"
        is_complete = info_file.exists() and tasks_file.exists()
        is_half_baked = dataset_root.exists() and not is_complete

        if is_half_baked:
            import shutil
            print(f"[lerobot] 불완전한 기존 폴더 정리: {dataset_root}")
            shutil.rmtree(dataset_root)

        if is_complete:
            print(f"[lerobot] 기존 데이터셋 로드: {dataset_root}")
            self.dataset = LeRobotDataset(repo_id=cfg.repo_id, root=dataset_root)
            self.dataset.start_image_writer(num_processes=0, num_threads=4)
        else:
            print(f"[lerobot] 신규 생성: {dataset_root}")
            self.dataset = LeRobotDataset.create(
                repo_id=cfg.repo_id,
                fps=cfg.hz,
                features=features,
                root=dataset_root,
                robot_type="franka_fr3",
                use_videos=True,
                image_writer_threads=4,
            )

        # ── 워커 상태 ─────────────────────────────────────────────────
        self._queue: queue.Queue = queue.Queue()
        self._state_lock = threading.Lock()
        # 완료 저장된 에피소드 수 (워커가 save 성공 시 증가)
        self._num_saved = int(self.dataset.num_episodes)
        # 워커가 현재 처리 중인 저장 작업 수 (UI 표시용)
        self._pending_saves = 0
        self._worker = threading.Thread(
            target=self._run_worker, daemon=True, name="lerobot-recorder"
        )
        self._worker.start()

    # ── 워커 루프 ────────────────────────────────────────────────────
    def _run_worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            kind = item[0]
            try:
                if kind == "frame":
                    _, frame = item
                    self.dataset.add_frame(frame)
                elif kind == self._CMD_SAVE:
                    self.dataset.save_episode()
                    with self._state_lock:
                        self._num_saved = int(self.dataset.num_episodes)
                        self._pending_saves -= 1
                elif kind == self._CMD_DISCARD:
                    self.dataset.clear_episode_buffer(delete_images=True)
                    with self._state_lock:
                        self._pending_saves -= 1
                elif kind == self._CMD_FINALIZE:
                    done_evt: threading.Event = item[1]
                    self.dataset.finalize()
                    done_evt.set()
                    return
            except Exception as e:
                print(f"[recorder/{kind}] 실패: {e}")
                if kind in (self._CMD_SAVE, self._CMD_DISCARD):
                    with self._state_lock:
                        self._pending_saves -= 1
            finally:
                self._queue.task_done()

    # ── 활성 task ───────────────────────────────────────────────────
    @property
    def active_task(self) -> str:
        return self._active_task

    def set_active_task(self, task_id: str) -> None:
        if task_id not in self.cfg.task_pools:
            raise ValueError(f"unknown task_id: {task_id}")
        self._active_task = task_id

    # ── 메인 스레드에서 호출 ─────────────────────────────────────────
    def add(self, state: np.ndarray, action: np.ndarray,
            images_bgr: dict[str, np.ndarray]) -> None:
        pool = self.cfg.task_pools[self._active_task]
        frame: dict = {
            "action": action.astype(np.float32),
            "observation.state": state.astype(np.float32),
            "task": self._rng.choice(pool),
        }
        for name, bgr in images_bgr.items():
            frame[f"observation.images.{name}"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._queue.put(("frame", frame))

    def save(self) -> None:
        with self._state_lock:
            self._pending_saves += 1
        self._queue.put((self._CMD_SAVE,))

    def discard(self) -> None:
        with self._state_lock:
            self._pending_saves += 1
        self._queue.put((self._CMD_DISCARD,))

    def finalize(self) -> None:
        """모든 pending 작업 flush 후 finalize. 블로킹."""
        done = threading.Event()
        self._queue.put((self._CMD_FINALIZE, done))
        done.wait()
        self._worker.join(timeout=5.0)

    @property
    def num_saved(self) -> int:
        with self._state_lock:
            return self._num_saved

    @property
    def pending_saves(self) -> int:
        with self._state_lock:
            return self._pending_saves


# ─── 메인 루프 ───────────────────────────────────────────────────────────────

class DataCollector:
    def __init__(self, cfg: CollectConfig) -> None:
        self.cfg = cfg
        self.scalar_buf = LatestBuffer()   # topic → ROS msg
        self.image_buf = LatestBuffer()    # cam_name → BGR ndarray (reshape 완료)

        domains: dict[int, DomainSubscriber] = {}

        def _get_domain(did: int) -> DomainSubscriber:
            if did not in domains:
                domains[did] = DomainSubscriber(did, f"data_collect_d{did}")
            return domains[did]

        # 스칼라 토픽
        for spec in (cfg.robot_action, cfg.gripper_action,
                     cfg.robot_state, cfg.gripper_state):
            dom = _get_domain(spec.domain_id)
            topic = spec.topic
            dom.subscribe(topic, spec.msg_type,
                          lambda msg, t=topic: self.scalar_buf.set(t, msg))

        # 카메라 토픽
        for cam in cfg.cameras:
            dom = _get_domain(cam.domain_id)
            cb = _make_image_callback(self.image_buf, cam.name, cam.rotate, cfg.img_shape)
            dom.subscribe(cam.topic, cam.msg_type, cb)

        self._domains = domains
        self.recorder = EpisodeRecorder(cfg)

        # 녹화 상태
        self._recording = False
        self._rec_start_ts: float | None = None
        self._frame_count = 0

    def start(self) -> None:
        for d in self._domains.values():
            d.start()

    def shutdown(self) -> None:
        for d in self._domains.values():
            d.shutdown()

    # ── state/action 벡터 조립 ──
    def _state_vec(self, scalar_snap) -> np.ndarray | None:
        rs = scalar_snap.get(self.cfg.robot_state.topic)
        gs = scalar_snap.get(self.cfg.gripper_state.topic)
        if rs is None or gs is None:
            return None
        arm = np.asarray(rs[0].position, dtype=np.float32)
        grip = np.asarray(gs[0].position, dtype=np.float32)
        if arm.size < 7 or grip.size < 1:
            return None
        return np.concatenate([arm[:7], grip[:1]])

    def _action_vec(self, scalar_snap) -> np.ndarray | None:
        ra = scalar_snap.get(self.cfg.robot_action.topic)
        ga = scalar_snap.get(self.cfg.gripper_action.topic)
        if ra is None or ga is None:
            return None
        arm = np.asarray(ra[0].position, dtype=np.float32)
        grip = float(ga[0].data)
        if arm.size < 7:
            return None
        return np.concatenate([arm[:7], np.array([grip], dtype=np.float32)])

    # ── 키 입력 ──
    # waitKeyEx 기준 화살표 키코드 (Linux/X11; 다른 플랫폼 호환을 위해 다중 등록)
    _KEY_LEFT = {65361, 81, 2424832}   # X11, waitKey&0xFF 하위호환, Windows
    _KEY_RIGHT = {65363, 83, 2555904}

    def _shift_active_task(self, delta: int) -> None:
        task_ids = self.cfg.task_ids
        if not task_ids:
            return
        cur = self.recorder.active_task
        idx = task_ids.index(cur) if cur in task_ids else 0
        new_idx = (idx + delta) % len(task_ids)
        new_task = task_ids[new_idx]
        self.recorder.set_active_task(new_task)
        print(f"[task] {cur} → {new_task}  ({new_idx + 1}/{len(task_ids)})")

    def _handle_key(self, key: int) -> bool:
        """True를 반환하면 루프 종료."""
        ascii_key = key & 0xFF

        if key in self._KEY_LEFT or key in self._KEY_RIGHT:
            if self._recording:
                print("[task] 녹화 중에는 task 전환 불가")
            else:
                self._shift_active_task(-1 if key in self._KEY_LEFT else +1)
            return False

        if ascii_key == ord("s"):
            if self._recording:
                print("[key:s] 이미 녹화 중")
            else:
                self._recording = True
                self._rec_start_ts = time.time()
                self._frame_count = 0
                print(f"[key:s] 녹화 시작 (warmup)  task={self.recorder.active_task}")
        elif ascii_key == ord(" ") or ascii_key == 32:
            if self._recording and self._frame_count > 0:
                print(f"[key:space] 저장 요청 전송 ({self._frame_count} frames, async)")
                self.recorder.save()
            elif self._recording:
                print("[key:space] 저장할 프레임 없음 → discard")
                self.recorder.discard()
            else:
                print("[key:space] 녹화 중이 아님")
            self._recording = False
            self._rec_start_ts = None
            self._frame_count = 0
        elif ascii_key == ord("r"):
            if self._recording:
                print("[key:r] 현재 에피소드 버림")
                self.recorder.discard()
            self._recording = False
            self._rec_start_ts = None
            self._frame_count = 0
        elif ascii_key == ord("q"):
            if self._recording:
                print("[key:q] 녹화 중 종료 → discard")
                self.recorder.discard()
            print("[key:q] finalize (pending 저장 완료 대기 중...)")
            self.recorder.finalize()
            print("[key:q] finalize 완료")
            return True
        return False

    # ── 메인 루프 ──
    def run(self) -> None:
        cfg = self.cfg
        period = 1.0 / cfg.hz
        cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
        print("[data_collect] ready — s: start, space: save, r: discard, q: quit,"
              "  ←/→: prev/next task")

        while True:
            tick = time.time()

            scalar_snap = self.scalar_buf.snapshot()
            image_snap = self.image_buf.snapshot()
            scalar_ts = {k: v[1] for k, v in scalar_snap.items()}
            cam_images = {k: v[0] for k, v in image_snap.items()}
            cam_ts = {k: v[1] for k, v in image_snap.items()}

            state = self._state_vec(scalar_snap)
            action = self._action_vec(scalar_snap)

            # 상태 문자열
            if self._recording and self._rec_start_ts is not None:
                elapsed = time.time() - self._rec_start_ts
                if elapsed < cfg.warmup_sec:
                    status = f"WARMUP {elapsed:.1f}/{cfg.warmup_sec:.1f}s"
                else:
                    status = "REC"
            else:
                elapsed = 0.0
                status = "IDLE"

            # 저장
            ready = (
                self._recording
                and self._rec_start_ts is not None
                and (time.time() - self._rec_start_ts) >= cfg.warmup_sec
                and state is not None
                and action is not None
                and len(cam_images) == len(cfg.cameras)
            )
            if ready:
                try:
                    self.recorder.add(state, action, cam_images)
                    self._frame_count += 1
                except Exception as e:
                    print(f"[add_frame 실패] {e}")

            # 시각화
            saved_eps = self.recorder.num_saved
            pending = self.recorder.pending_saves
            task_ids = cfg.task_ids
            active = self.recorder.active_task
            task_idx = task_ids.index(active) if active in task_ids else 0
            canvas = render(
                cfg, cam_images, cam_ts, scalar_ts,
                state, action, status, elapsed, self._frame_count,
                saved_episodes=saved_eps,
                current_episode=saved_eps + pending,
                active_task=active,
                task_index=task_idx,
                task_total=len(task_ids),
                pending_saves=pending,
            )
            cv2.imshow(_WINDOW_NAME, canvas)
            key = cv2.waitKeyEx(1)
            if key != -1 and self._handle_key(key):
                break

            # hz 유지
            sleep = period - (time.time() - tick)
            if sleep > 0:
                time.sleep(sleep)

        cv2.destroyAllWindows()


# ─── entry ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ROS2 → LeRobot data collector")
    parser.add_argument(
        "--config", type=Path,
        default=Path("/workspace/m.ax/config/proj_gt_kitting_config.yaml"),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"[config] hz={cfg.hz} warmup={cfg.warmup_sec}s img={cfg.img_shape} "
          f"repo_id={cfg.repo_id}")
    print("[config] tasks:")
    for tid, pool in cfg.task_pools.items():
        print(f"  - {tid} ({len(pool)} prompts)")
    print(f"[config] cams: {[c.name for c in cfg.cameras]}")

    collector = DataCollector(cfg)
    collector.start()
    try:
        collector.run()
    except KeyboardInterrupt:
        print("\n[interrupt] shutting down")
    finally:
        collector.shutdown()


if __name__ == "__main__":
    main()
