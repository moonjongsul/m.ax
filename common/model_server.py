"""ModelServer — VLA 모델 로드 + 추론 서버

ZMQ 포트:
  SUB  :5570  ← CommServer robot/gripper PUB (pickle JointState/Float32)
  SUB  :5572  ← CommServer camera PUB (lerobot JSON)
  PUSH :5590  → CommServer Arbiter (추론 액션 → ROS publish)
  PUB  :5592  → Streamlit (모델 상태 알림)
  PULL :5593  ← Streamlit (start+prompt / stop 명령)

모델 상태:
  idle      : checkpoint 없음 또는 로드 전
  loading   : from_pretrained 진행 중
  ready     : 로드 완료, 추론 대기
  running   : 추론 루프 실행 중
  error     : 로드 실패

제어 메시지 포맷 (Streamlit → ModelServer PULL :5593):
  [b"start", prompt_bytes]   추론 시작
  [b"stop"]                  추론 정지

상태 알림 포맷 (ModelServer PUB :5592 → Streamlit):
  [b"model_state", b"<state>"]            상태 변경
  [b"model_state", b"error:<message>"]   에러 상세
"""

import base64
import json
import logging
import pickle
import threading
import time
from typing import Literal

import cv2
import numpy as np
import torch
import zmq

logger = logging.getLogger(__name__)

ZMQ_HOST           = "localhost"
ZMQ_ROBOT_PUB_PORT = 5570   # CommServer robot/gripper PUB
ZMQ_CAM_PUB_PORT   = 5572   # CommServer camera PUB
ZMQ_CTRL_PUSH_PORT = 5590   # CommServer Arbiter PULL
ZMQ_STATE_PUB_PORT = 5592   # ModelServer 상태 PUB
ZMQ_CMD_PULL_PORT  = 5593   # ModelServer 명령 PULL

ModelState = Literal["idle", "loading", "ready", "running", "error"]

POLICY_MAP = {
    "pi0.5": "lerobot.policies.pi05.modeling_pi05.PI05Policy",
    "smolvla": "lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy",
}


def _load_policy_class(policy_name: str):
    """policy 이름으로 Policy 클래스 동적 임포트."""
    path = POLICY_MAP.get(policy_name)
    if path is None:
        raise ValueError(f"Unknown policy '{policy_name}'. Available: {list(POLICY_MAP)}")
    module_path, class_name = path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class ModelServer:
    """VLA 모델 로드 + 추론 루프 + ZMQ 통신."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._model_cfg = cfg.model

        self._state: ModelState = "idle"
        self._state_lock = threading.Lock()

        self._policy = None
        self._preprocessor = None
        self._postprocessor = None

        self._prompt: str = str(self._model_cfg.get("task_prompt") or "")
        self._inference_stop = threading.Event()
        self._running = False

        self._zmq_ctx = zmq.Context()
        self._state_pub = None
        self._cmd_pull  = None
        self._ctrl_push = None
        self._robot_sub = None
        self._cam_sub   = None

        # 최신 관측값 버퍼 (추론 루프에서 읽음)
        self._obs_lock = threading.Lock()
        self._latest_joints: np.ndarray | None = None      # 8-dim (7 joint + 1 gripper)
        self._latest_images: dict[str, np.ndarray] | None = None  # name → RGB (H,W,3)

    # ─── 상태 관리 ────────────────────────────────────────────────────────────

    def _set_state(self, state: ModelState, detail: str = ""):
        with self._state_lock:
            self._state = state
        msg = f"error:{detail}" if state == "error" and detail else state
        if self._state_pub:
            self._state_pub.send_multipart([b"model_state", msg.encode()])
        logger.info(f"[ModelServer] state → {msg}")

    @property
    def state(self) -> ModelState:
        with self._state_lock:
            return self._state

    # ─── ZMQ 소켓 초기화 ──────────────────────────────────────────────────────

    def _init_sockets(self):
        # 상태 PUB
        self._state_pub = self._zmq_ctx.socket(zmq.PUB)
        self._state_pub.bind(f"tcp://*:{ZMQ_STATE_PUB_PORT}")

        # 명령 PULL
        self._cmd_pull = self._zmq_ctx.socket(zmq.PULL)
        self._cmd_pull.bind(f"tcp://*:{ZMQ_CMD_PULL_PORT}")
        self._cmd_pull.setsockopt(zmq.RCVTIMEO, 200)

        # Arbiter PUSH (액션 전송)
        self._ctrl_push = self._zmq_ctx.socket(zmq.PUSH)
        self._ctrl_push.connect(f"tcp://{ZMQ_HOST}:{ZMQ_CTRL_PUSH_PORT}")

        # robot/gripper SUB
        self._robot_sub = self._zmq_ctx.socket(zmq.SUB)
        self._robot_sub.setsockopt(zmq.RCVTIMEO, 100)
        self._robot_sub.setsockopt(zmq.RCVHWM, 1)
        self._robot_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._robot_sub.connect(f"tcp://{ZMQ_HOST}:{ZMQ_ROBOT_PUB_PORT}")

        # camera SUB
        self._cam_sub = self._zmq_ctx.socket(zmq.SUB)
        self._cam_sub.setsockopt(zmq.RCVTIMEO, 500)
        self._cam_sub.setsockopt(zmq.CONFLATE, 1)   # 항상 최신 프레임만 유지
        self._cam_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._cam_sub.connect(f"tcp://{ZMQ_HOST}:{ZMQ_CAM_PUB_PORT}")

    # ─── 모델 로드 ────────────────────────────────────────────────────────────

    def _load(self):
        checkpoint = str(self._model_cfg.get("checkpoint") or "").strip()
        if not checkpoint:
            self._set_state("idle")
            logger.warning("[ModelServer] checkpoint가 비어있어 모델 로드를 건너뜁니다.")
            return

        # 로컬 경로(./ , .checkpoints/, 상대경로 등)는 절대경로로 변환.
        # HuggingFace repo ID(예: 'lerobot/pi05')는 '/'가 1개이고 '.'나 '/'로 시작하지 않음.
        from pathlib import Path
        ckpt_path = Path(checkpoint)
        looks_like_path = (
            checkpoint.startswith(("./", "../", ".", "/"))
            or ckpt_path.exists()
        )
        if looks_like_path:
            checkpoint = str(ckpt_path.resolve())

        self._set_state("loading")
        try:
            from lerobot.policies.factory import make_pre_post_processors

            policy_name = str(self._model_cfg.get("policy") or "pi0.5")
            device      = str(self._model_cfg.get("device") or "cuda")
            dtype_str   = str(self._model_cfg.get("dtype") or "bfloat16")
            dtype       = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32

            logger.info(f"[ModelServer] Loading {policy_name} from {checkpoint} ...")
            PolicyClass = _load_policy_class(policy_name)
            policy = PolicyClass.from_pretrained(checkpoint)
            policy.to(device=device, dtype=dtype)
            policy.eval()

            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=policy.config,
                pretrained_path=checkpoint,
            )

            self._policy       = policy
            self._preprocessor = preprocessor
            self._postprocessor = postprocessor
            self._set_state("ready")
            logger.info("[ModelServer] 모델 로드 완료.")

        except Exception as e:
            self._set_state("error", str(e))
            logger.error(f"[ModelServer] 모델 로드 실패: {e}")

    # ─── 관측값 수집 스레드 ───────────────────────────────────────────────────

    def _robot_recv_loop(self):
        """robot/gripper SUB → _latest_joints 갱신.

        JointState 타입 토픽만 처리. PoseStamped 등 다른 타입은 무시.
        추론 입력은 /joint_states (7-DOF) + gripper (1-DOF) = 8-dim.
        """
        joint_buf: list[float] | None = None
        gripper_buf: float | None = None

        # yaml에서 JointState 타입인 토픽만 추출
        env = self.cfg.env
        robot_joint_topics: set[str] = set()
        gripper_sub_topics: set[str] = set()

        robot = env.get("robot")
        if robot:
            for topic, info in (robot.topic.get("subscribe") or {}).items():
                if "JointState" in str(info.get("type", "")):
                    robot_joint_topics.add(topic)

        gripper = env.get("gripper")
        if gripper:
            for topic, info in (gripper.topic.get("subscribe") or {}).items():
                if "JointState" in str(info.get("type", "")):
                    gripper_sub_topics.add(topic)

        # 추론용 robot joint는 첫 번째 JointState 토픽 사용
        primary_robot_topic = next(iter(robot_joint_topics), None)

        while self._running:
            try:
                parts = self._robot_sub.recv_multipart(zmq.NOBLOCK)
                if len(parts) != 2:
                    continue
                try:
                    topic = parts[0].decode("utf-8")
                except UnicodeDecodeError:
                    continue

                # JointState 토픽이 아니면 무시
                if topic not in robot_joint_topics and topic not in gripper_sub_topics:
                    continue

                msg = pickle.loads(raw := parts[1])

                if topic == primary_robot_topic:
                    joint_buf = list(msg.position[:7])
                elif topic in gripper_sub_topics:
                    if msg.position:
                        gripper_buf = float(msg.position[0])

                if joint_buf is not None and gripper_buf is not None:
                    state = np.array(joint_buf + [gripper_buf], dtype=np.float32)
                    with self._obs_lock:
                        self._latest_joints = state

            except zmq.Again:
                time.sleep(0.005)
            except Exception as e:
                logger.warning(f"[ModelServer] robot recv error: {e}")

    def _camera_recv_loop(self):
        """camera SUB (lerobot JSON) → _latest_images 갱신."""
        while self._running:
            try:
                msg_str = self._cam_sub.recv_string()
                data = json.loads(msg_str)
                images_b64 = data.get("images", {})
                images = {}
                for name, b64 in images_b64.items():
                    img_bytes = base64.b64decode(b64)
                    arr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        images[name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if images:
                    with self._obs_lock:
                        self._latest_images = images
            except zmq.Again:
                time.sleep(0.01)
            except Exception as e:
                logger.warning(f"[ModelServer] camera recv error: {e}")

    # ─── 추론 루프 ────────────────────────────────────────────────────────────

    def _build_observation(self, joints: np.ndarray, images: dict[str, np.ndarray],
                           prompt: str) -> dict:
        obs = {
            "observation.state": torch.from_numpy(joints).unsqueeze(0),
            "task": prompt,
        }
        for cam_name, img_rgb in images.items():
            # (H,W,3) uint8 RGB → (1,3,H,W) float [0,1]
            t = torch.from_numpy(img_rgb).float() / 255.0
            obs[f"observation.images.{cam_name}"] = t.permute(2, 0, 1).unsqueeze(0)
        return obs

    @staticmethod
    def _make_stamp():
        from builtin_interfaces.msg import Time
        t = time.time()
        stamp = Time()
        stamp.sec     = int(t)
        stamp.nanosec = int((t % 1) * 1e9)
        return stamp

    def _publish_action(self, action: np.ndarray):
        """추론 액션(8-dim) → Arbiter PUSH → ROS publish."""
        env = self.cfg.env

        # robot publish 토픽
        robot = env.get("robot")
        if robot:
            pub_topics = list((robot.topic.get("publish") or {}).keys())
            joint_names = list(robot.get("joint_list") or [])
            if pub_topics and joint_names:
                from sensor_msgs.msg import JointState
                from std_msgs.msg import Header
                n = len(joint_names)
                msg = JointState()
                msg.header       = Header()
                msg.header.stamp = self._make_stamp()
                msg.name         = joint_names
                msg.position     = action[:n].tolist()
                msg.velocity     = [0.0] * n
                msg.effort       = [0.0] * n
                raw = pickle.dumps(msg)
                self._ctrl_push.send_multipart([b"model", pub_topics[0].encode(), raw])

        # gripper publish 토픽
        gripper = env.get("gripper")
        if gripper:
            pub_topics = list((gripper.topic.get("publish") or {}).keys())
            if pub_topics:
                from std_msgs.msg import Float32
                msg = Float32()
                msg.data = float(action[7])
                raw = pickle.dumps(msg)
                self._ctrl_push.send_multipart([b"model", pub_topics[0].encode(), raw])

    def _inference_loop(self, prompt: str):
        """추론 메인 루프 — _inference_stop 이벤트까지 반복."""
        self._policy.reset()
        logger.info(f"[ModelServer] 추론 시작. prompt='{prompt}'")

        while not self._inference_stop.is_set():
            with self._obs_lock:
                joints = self._latest_joints
                images = self._latest_images

            if joints is None or images is None:
                time.sleep(0.01)
                continue

            obs = self._build_observation(joints, images, prompt)

            try:
                with torch.inference_mode():
                    preprocessed = self._preprocessor(obs)
                    action = self._policy.select_action(preprocessed)
                    action_out = self._postprocessor(action)
                action_np = action_out.cpu().numpy().squeeze(0)  # (8,)
                self._publish_action(action_np)
            except Exception as e:
                logger.error(f"[ModelServer] 추론 오류: {e}")
                self._set_state("error", str(e))
                break

        self._set_state("ready")
        logger.info("[ModelServer] 추론 정지.")

    # ─── 명령 수신 루프 ───────────────────────────────────────────────────────

    def _command_loop(self):
        """Streamlit으로부터 start/stop 명령 수신."""
        while self._running:
            try:
                parts = self._cmd_pull.recv_multipart()
            except zmq.Again:
                continue

            if not parts:
                continue

            cmd = parts[0].decode()

            if cmd == "start":
                if self.state != "ready":
                    logger.warning(f"[ModelServer] start 무시 (state={self.state})")
                    continue
                prompt = parts[1].decode() if len(parts) > 1 else self._prompt
                self._prompt = prompt
                self._inference_stop.clear()
                self._set_state("running")
                threading.Thread(
                    target=self._inference_loop,
                    args=(prompt,),
                    daemon=True,
                    name="inference-loop",
                ).start()

            elif cmd == "stop":
                if self.state == "running":
                    self._inference_stop.set()
                    # 상태는 _inference_loop 종료 시 "ready"로 변경

            elif cmd == "reload":
                # checkpoint 경로 변경 후 재로드
                if len(parts) > 1:
                    new_ckpt = parts[1].decode()
                    self._model_cfg.checkpoint = new_ckpt
                if self.state == "running":
                    self._inference_stop.set()
                    time.sleep(0.5)
                threading.Thread(target=self._load, daemon=True, name="model-load").start()

    # ─── 공개 인터페이스 ──────────────────────────────────────────────────────

    def start(self):
        """소켓 초기화 → 모델 로드 → 수신/명령 스레드 기동."""
        self._running = True
        self._init_sockets()

        # 모델 로드 (백그라운드)
        threading.Thread(target=self._load, daemon=True, name="model-load").start()

        # 관측값 수집 스레드
        threading.Thread(target=self._robot_recv_loop, daemon=True, name="robot-recv").start()
        threading.Thread(target=self._camera_recv_loop, daemon=True, name="cam-recv").start()

        # 명령 수신 스레드
        threading.Thread(target=self._command_loop, daemon=True, name="model-cmd").start()

        logger.info(f"[ModelServer] started. STATE PUB:{ZMQ_STATE_PUB_PORT}, "
                    f"CMD PULL:{ZMQ_CMD_PULL_PORT}")

    def shutdown(self):
        self._running = False
        self._inference_stop.set()
        time.sleep(0.5)  # 스레드들이 recv 타임아웃 후 _running 체크하고 종료할 시간
        for sock in (self._state_pub, self._cmd_pull, self._ctrl_push,
                     self._robot_sub, self._cam_sub):
            if sock:
                sock.setsockopt(zmq.LINGER, 0)
                sock.close()
        self._zmq_ctx.term()
        logger.info("[ModelServer] shutdown complete.")
