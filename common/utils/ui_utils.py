"""common/utils/ui_utils.py

Streamlit UI 컴포넌트 모음.
각 render_*() 함수는 placeholder dict를 반환하고,
update_*() 함수는 해당 placeholder에 최신 데이터를 밀어 넣는다.

ZMQ 포트:
  SUB :5570  — robot/gripper (pickle, per domain_id=0)
  SUB :5572  — camera (lerobot JSON)
  PUSH :5590 — 제어 명령 → Arbiter
  SUB :5591  — 제어권 알림 (granted/revoked)
  SUB :5592  — 모델 상태 알림
  PUSH :5593 — 모델 명령 (start/stop/reload)
"""

import base64
import json
import pickle
import threading
from typing import Any

import cv2
import numpy as np
import streamlit as st
import zmq
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

# ─── ZMQ 설정 ────────────────────────────────────────────────────────────────

ZMQ_HOST           = "localhost"
ZMQ_ROBOT_PUB_PORT = 5570
ZMQ_CAM_PUB_PORT   = 5572
ZMQ_CTRL_PUSH_PORT = 5590
ZMQ_CTRL_PUB_PORT  = 5591
ZMQ_STATE_PUB_PORT = 5592
ZMQ_MODEL_CMD_PORT = 5593

CONTROL_SOURCE = "web"

# ─── 공통 헬퍼 ───────────────────────────────────────────────────────────────

ROTATE_MAP = {
    90:  cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


_COMPACT_CSS = """
<style>
  /* divider 여백 축소 */
  hr { margin: 0.4rem 0 !important; }
  /* subheader 여백 축소 (h1 페이지 제목은 건드리지 않음) */
  h2, h3 { margin-top: 0.4rem !important; margin-bottom: 0.2rem !important; }
  /* caption 여백 */
  .stCaption { margin-bottom: 0.1rem !important; }
  /* 각 element 사이 간격 축소 */
  .stVerticalBlock { gap: 0.3rem !important; }
</style>
"""


def _inject_compact_css():
    st.html(_COMPACT_CSS)


def _to_html_img(img_bgr: np.ndarray) -> str:
    """BGR numpy 배열 → base64 JPEG HTML img 태그."""
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf).decode()
    return f'<img src="data:image/jpeg;base64,{b64}" style="width:100%;border-radius:4px;">'


def _no_signal_html() -> str:
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(img, "No Signal", (55, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (160, 160, 160), 2)
    # no-signal은 BGR이지만 색상 무관하므로 그대로 인코딩
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf).decode()
    return f'<img src="data:image/jpeg;base64,{b64}" style="width:100%;border-radius:4px;">'


# ─── ZMQ 소켓 (단일 Context 공유) ────────────────────────────────────────────

@st.cache_resource
def _zmq_context() -> zmq.Context:
    return zmq.Context()


@st.cache_resource
def _robot_sub_socket() -> zmq.Socket:
    ctx = _zmq_context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVTIMEO, 100)
    sock.setsockopt(zmq.RCVHWM, 1)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.connect(f"tcp://{ZMQ_HOST}:{ZMQ_ROBOT_PUB_PORT}")
    return sock


@st.cache_resource
def _camera_sub_socket() -> zmq.Socket:
    ctx = _zmq_context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVTIMEO, 100)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.connect(f"tcp://{ZMQ_HOST}:{ZMQ_CAM_PUB_PORT}")
    return sock


@st.cache_resource
def _ctrl_push_socket() -> zmq.Socket:
    ctx = _zmq_context()
    sock = ctx.socket(zmq.PUSH)
    sock.setsockopt(zmq.SNDHWM, 10)
    sock.connect(f"tcp://{ZMQ_HOST}:{ZMQ_CTRL_PUSH_PORT}")
    return sock


@st.cache_resource
def _model_cmd_socket() -> zmq.Socket:
    ctx = _zmq_context()
    sock = ctx.socket(zmq.PUSH)
    sock.setsockopt(zmq.SNDHWM, 5)
    sock.connect(f"tcp://{ZMQ_HOST}:{ZMQ_MODEL_CMD_PORT}")
    return sock


@st.cache_resource
def _ctrl_state() -> dict:
    return {"owner": None, "lock": threading.Lock()}


@st.cache_resource
def _model_state() -> dict:
    return {"state": "idle", "detail": "", "lock": threading.Lock()}


@st.cache_resource
def _listeners_started() -> dict:
    """제어권 + 모델 상태 리스너를 한 번만 기동하는 sentinel."""
    ctx = _zmq_context()

    # 제어권 리스너
    ctrl_sock = ctx.socket(zmq.SUB)
    ctrl_sock.setsockopt(zmq.RCVTIMEO, 300)
    ctrl_sock.setsockopt_string(zmq.SUBSCRIBE, "control")
    ctrl_sock.connect(f"tcp://{ZMQ_HOST}:{ZMQ_CTRL_PUB_PORT}")

    def _ctrl_listen():
        state = _ctrl_state()
        while True:
            try:
                _, msg = ctrl_sock.recv_multipart()
                event, source = msg.decode().split(":", 1)
                with state["lock"]:
                    if event == "granted":
                        state["owner"] = source
                    elif event == "revoked" and state["owner"] == source:
                        state["owner"] = None
            except zmq.Again:
                pass
            except Exception:
                pass

    # 모델 상태 리스너
    model_sock = ctx.socket(zmq.SUB)
    model_sock.setsockopt(zmq.RCVTIMEO, 300)
    model_sock.setsockopt_string(zmq.SUBSCRIBE, "model_state")
    model_sock.connect(f"tcp://{ZMQ_HOST}:{ZMQ_STATE_PUB_PORT}")

    def _model_listen():
        ms = _model_state()
        while True:
            try:
                _, msg = model_sock.recv_multipart()
                decoded = msg.decode()
                with ms["lock"]:
                    if decoded.startswith("error:"):
                        ms["state"] = "error"
                        ms["detail"] = decoded[6:]
                    else:
                        ms["state"] = decoded
                        ms["detail"] = ""
                    ms["_received"] = True
            except zmq.Again:
                pass
            except Exception:
                pass

    threading.Thread(target=_ctrl_listen,  daemon=True, name="ctrl-listener").start()
    threading.Thread(target=_model_listen, daemon=True, name="model-listener").start()
    return {"started": True}


def _start_listeners():
    _listeners_started()


# ─── 제어 메시지 전송 ─────────────────────────────────────────────────────────

def _now_stamp():
    """현재 시각을 ROS Header stamp로 반환 (rclpy.clock 없이 time.time() 사용)."""
    import time
    from builtin_interfaces.msg import Time
    t = time.time()
    stamp = Time()
    stamp.sec     = int(t)
    stamp.nanosec = int((t % 1) * 1e9)
    return stamp


@st.cache_resource
def _burst_state() -> dict:
    """활성 burst를 관리하는 공유 상태. 동시에 최대 1개 burst만 실행."""
    return {"cancel": threading.Event(), "lock": threading.Lock()}


def _burst_send(send_fn, hz: int = 100, duration: float = 1.0):
    """send_fn()을 hz 주기로 duration 초 동안 반복 발행.

    기존 burst가 실행 중이면 취소하고 새 burst 시작.
    send_fn은 매 호출마다 최신 stamp로 메시지를 만들어 전송하는 callable.
    """
    import time as _time
    bs = _burst_state()

    with bs["lock"]:
        # 기존 burst 취소
        bs["cancel"].set()
        # 새 cancel event 생성
        cancel = threading.Event()
        bs["cancel"] = cancel

    def _run():
        interval = 1.0 / hz
        end_time = _time.monotonic() + duration
        while _time.monotonic() < end_time and not cancel.is_set():
            send_fn()
            _time.sleep(interval)

    threading.Thread(target=_run, daemon=True, name="pose-burst").start()


def _send_joint_pose(pub_topic: str, positions: list[float], names: list[str]):
    """joint pose → 100Hz × 1초 burst 발행."""
    def _send():
        msg = JointState()
        msg.header        = Header()
        msg.header.stamp  = _now_stamp()
        msg.name          = names
        msg.position      = positions
        msg.velocity      = [0.0] * len(positions)
        msg.effort        = [0.0] * len(positions)
        raw = pickle.dumps(msg)
        _ctrl_push_socket().send_multipart([
            CONTROL_SOURCE.encode(),
            pub_topic.encode(),
            raw,
        ])
    _burst_send(_send)


def _send_float_pose(pub_topic: str, value: float):
    """float pose → 100Hz × 1초 burst 발행."""
    from std_msgs.msg import Float32

    def _send():
        msg = Float32()
        msg.data = value
        raw = pickle.dumps(msg)
        _ctrl_push_socket().send_multipart([
            CONTROL_SOURCE.encode(),
            pub_topic.encode(),
            raw,
        ])
    _burst_send(_send)


def send_model_start(prompt: str):
    _model_cmd_socket().send_multipart([b"start", prompt.encode()])


def send_model_stop():
    _model_cmd_socket().send_multipart([b"stop"])


def send_model_reload(checkpoint: str):
    _model_cmd_socket().send_multipart([b"reload", checkpoint.encode()])


# ─── Camera ──────────────────────────────────────────────────────────────────

@st.cache_resource
def _cam_frame_buffer() -> dict:
    """카메라 이름 → 최신 BGR 프레임 버퍼 (None = 수신 전)."""
    return {}


def _recv_latest_camera() -> dict[str, np.ndarray] | None:
    """lerobot JSON 포맷에서 최신 프레임 dict 수신. 없으면 None.

    CommServer가 BGR→RGB 변환 후 JPEG 인코딩하므로,
    cv2.imdecode 후 별도 변환 없이 RGB 배열 그대로 반환.
    """
    sock = _camera_sub_socket()
    latest = None
    try:
        while True:
            msg_str = sock.recv_string(zmq.NOBLOCK)
            data = json.loads(msg_str)
            images_b64 = data.get("images", {})
            frames = {}
            for name, b64 in images_b64.items():
                arr = np.frombuffer(base64.b64decode(b64), np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # → BGR
                if img is not None:
                    frames[name] = img  # BGR 그대로 유지 (_to_html_img가 BGR 입력)
            if frames:
                latest = frames
    except zmq.Again:
        pass
    return latest


def render_cameras(cfg) -> dict:
    _inject_compact_css()
    cameras: dict = cfg.env.get("camera") or {}
    cam_list = []
    for _, cam_cfg in cameras.items():
        cam_list.append({"name": cam_cfg.name})

    if not cam_list:
        return {}

    st.subheader("Cameras")
    n_cols = min(len(cam_list), 4)
    cols = st.columns(n_cols)

    placeholders: dict[str, Any] = {}
    for i, cam in enumerate(cam_list):
        with cols[i % n_cols]:
            st.markdown(f"**{cam['name']}**")
            placeholders[cam["name"]] = st.empty()

    return placeholders


def update_cameras(placeholders: dict):
    if not placeholders:
        return

    buf = _cam_frame_buffer()
    frames = _recv_latest_camera()
    if frames:
        buf.update(frames)

    for name, ph in placeholders.items():
        frame = buf.get(name)
        html = _to_html_img(frame) if frame is not None else _no_signal_html()
        ph.html(html)


# ─── Robot ───────────────────────────────────────────────────────────────────

def _recv_latest_robot(sub_topics: set[str]) -> dict[str, tuple[str, bytes]] | None:
    """robot sub 소켓에서 최신 토픽 수신. {topic: raw_bytes}"""
    sock = _robot_sub_socket()
    result: dict[str, bytes] = {}
    try:
        while True:
            parts = sock.recv_multipart(zmq.NOBLOCK)
            if len(parts) != 2:
                continue
            try:
                topic = parts[0].decode("utf-8")
            except UnicodeDecodeError:
                continue
            if topic in sub_topics:
                result[topic] = parts[1]
    except zmq.Again:
        pass
    return result if result else None


def render_robot(cfg) -> dict:
    robot = (cfg.env or {}).get("robot")
    if not robot:
        return {}

    sub_topics = list((robot.topic.get("subscribe") or {}).keys())
    pub_topics  = list((robot.topic.get("publish")   or {}).keys())
    all_topics  = sub_topics + pub_topics

    st.subheader("Robot")
    placeholders: dict[str, Any] = {}
    cols = st.columns(len(all_topics)) if all_topics else []
    for col, t in zip(cols, sub_topics):
        with col:
            st.caption(f"⬇ `{t}`")
            ph = st.empty()
            ph.caption("⏳ 대기 중...")
            placeholders[t] = ph
    for col, t in zip(cols[len(sub_topics):], pub_topics):
        with col:
            st.caption(f"⬆ `{t}`")
            ph = st.empty()
            ph.caption("— 발행 전 —")
            placeholders[f"pub:{t}"] = ph
    return placeholders


def _format_joint_state(msg) -> tuple[str, dict]:
    joints = {name: round(float(pos), 4)
              for name, pos in zip(msg.name, msg.position)}
    rows = "\n".join(f"- `{n}`: {v}" for n, v in joints.items())
    return rows, joints


def _format_pose_stamped(msg) -> str:
    p = msg.pose.position
    o = msg.pose.orientation
    return (
        f"**Position** &nbsp; "
        f"`x` {p.x:.4f} &nbsp; `y` {p.y:.4f} &nbsp; `z` {p.z:.4f}\n\n"
        f"**Orientation** &nbsp; "
        f"`x` {o.x:.4f} &nbsp; `y` {o.y:.4f} &nbsp; "
        f"`z` {o.z:.4f} &nbsp; `w` {o.w:.4f}"
    )


def update_robot(placeholders: dict, cfg):
    if not placeholders:
        return
    robot = (cfg.env or {}).get("robot")
    if not robot:
        return

    sub_topics = set((robot.topic.get("subscribe") or {}).keys())
    received = _recv_latest_robot(sub_topics)
    if not received:
        return

    for topic, raw in received.items():
        if topic not in placeholders:
            continue
        msg = pickle.loads(raw)
        # 실제 메시지 클래스로 분기 (yaml type 선언과 불일치 대비)
        msg_cls = type(msg).__name__

        if msg_cls == "PoseStamped":
            placeholders[topic].markdown(_format_pose_stamped(msg))
        elif msg_cls == "JointState":
            rows, joints = _format_joint_state(msg)
            placeholders[topic].markdown(rows)
            if "joint_states" in topic and "gripper" not in topic:
                st.session_state["_robot_current_joints"] = joints
                for pt in (robot.topic.get("publish") or {}):
                    pk = f"pub:{pt}"
                    if pk in placeholders:
                        placeholders[pk].markdown(rows)
        else:
            placeholders[topic].caption(f"지원하지 않는 메시지 타입: {msg_cls}")


# ─── Gripper ─────────────────────────────────────────────────────────────────

@st.cache_resource
def _gripper_sub_socket() -> zmq.Socket:
    """gripper는 robot과 같은 포트(5570)지만 별도 소켓으로 분리."""
    ctx = _zmq_context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVTIMEO, 100)
    sock.setsockopt(zmq.RCVHWM, 1)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.connect(f"tcp://{ZMQ_HOST}:{ZMQ_ROBOT_PUB_PORT}")
    return sock


def render_gripper(cfg) -> dict:
    gripper = (cfg.env or {}).get("gripper")
    if not gripper:
        return {}

    sub_topics = list((gripper.topic.get("subscribe") or {}).keys())
    pub_topics  = list((gripper.topic.get("publish")   or {}).keys())
    all_topics  = sub_topics + pub_topics

    st.subheader("Gripper")
    placeholders: dict[str, Any] = {}
    cols = st.columns(len(all_topics)) if all_topics else []
    for col, t in zip(cols, sub_topics):
        with col:
            st.caption(f"⬇ `{t}`")
            ph = st.empty()
            ph.caption("⏳ 대기 중...")
            placeholders[t] = ph
    for col, t in zip(cols[len(sub_topics):], pub_topics):
        with col:
            st.caption(f"⬆ `{t}`")
            ph = st.empty()
            ph.caption("— 발행 전 —")
            placeholders[f"pub:{t}"] = ph
    return placeholders


def update_gripper(placeholders: dict, cfg):
    if not placeholders:
        return
    gripper = (cfg.env or {}).get("gripper")
    if not gripper:
        return

    sub_topics = set((gripper.topic.get("subscribe") or {}).keys())
    sock = _gripper_sub_socket()
    result: dict[str, bytes] = {}
    try:
        while True:
            parts = sock.recv_multipart(zmq.NOBLOCK)
            if len(parts) != 2:
                continue
            try:
                topic = parts[0].decode("utf-8")
            except UnicodeDecodeError:
                continue
            if topic in sub_topics:
                result[topic] = parts[1]
    except zmq.Again:
        pass

    for topic, raw in result.items():
        if topic not in placeholders:
            continue
        msg = pickle.loads(raw)
        joints = {name: round(float(pos), 4)
                  for name, pos in zip(msg.name, msg.position)}
        rows = "\n".join(f"- `{n}`: {v}" for n, v in joints.items())
        placeholders[topic].markdown(rows)
        st.session_state["_gripper_current_joints"] = joints
        # pub 토픽 placeholder에 현재 값 표시
        for pt in (gripper.topic.get("publish") or {}):
            pk = f"pub:{pt}"
            if pk in placeholders:
                placeholders[pk].markdown(rows)


# ─── Pose 버튼 제어 ───────────────────────────────────────────────────────────

def render_pose_control(cfg, disabled: bool = False):
    """robot + gripper pose 버튼 렌더링."""
    robot   = (cfg.env or {}).get("robot")
    gripper = (cfg.env or {}).get("gripper")

    # ── Robot pose 버튼 ───────────────────────────────────────────────────
    if robot:
        poses: dict = dict(robot.get("pose") or {})
        pub_topics = list((robot.topic.get("publish") or {}).keys())

        if poses and pub_topics:
            pub_topic = pub_topics[0]
            # joint_list(yaml)를 최우선. 없으면 수신된 JointState에서 가져옴.
            joint_names = list(robot.get("joint_list") or [])
            if not joint_names:
                current: dict[str, float] = st.session_state.get(
                    "_robot_current_joints", {})
                joint_names = list(current.keys())
            joints_ready = bool(joint_names)

            st.markdown("**Robot Pose**")
            if not joints_ready:
                st.caption("⏳ 로봇 joint 상태 수신 대기 중...")
            MAX_PER_ROW = 6
            pose_items = list(poses.items())
            for row_start in range(0, len(pose_items), MAX_PER_ROW):
                row = pose_items[row_start:row_start + MAX_PER_ROW]
                cols = st.columns(len(row))
                for col, (pose_name, pose_vals) in zip(cols, row):
                    with col:
                        if st.button(
                            pose_name.capitalize(),
                            key=f"pose_robot_{pub_topic}_{pose_name}",
                            disabled=disabled or not joints_ready,
                            use_container_width=True,
                        ):
                            positions = [
                                float(pose_vals[i]) if i < len(pose_vals) else 0.0
                                for i in range(len(joint_names))
                            ]
                            _send_joint_pose(pub_topic, positions, joint_names)

    # ── Gripper pose 버튼 ─────────────────────────────────────────────────
    if gripper:
        poses: dict = dict(gripper.get("pose") or {})
        pub_topics = list((gripper.topic.get("publish") or {}).keys())
        pub_infos  = list((gripper.topic.get("publish") or {}).values())

        if poses and pub_topics:
            pub_topic = pub_topics[0]
            info = pub_infos[0]

            st.markdown("**Gripper Pose**")
            MAX_PER_ROW = 6
            pose_items = list(poses.items())
            for row_start in range(0, len(pose_items), MAX_PER_ROW):
                row = pose_items[row_start:row_start + MAX_PER_ROW]
                cols = st.columns(len(row))
                for col, (pose_name, pose_vals) in zip(cols, row):
                    with col:
                        if st.button(
                            pose_name.capitalize(),
                            key=f"pose_gripper_{pub_topic}_{pose_name}",
                            disabled=disabled,
                            use_container_width=True,
                        ):
                            if "Float32" in info.type:
                                _send_float_pose(pub_topic, float(pose_vals[0]))
                            else:
                                current_g = st.session_state.get(
                                    "_gripper_current_joints", {})
                                gnames = list(current_g.keys()) if current_g \
                                         else ["gripper_joint"]
                                gvals = [float(pose_vals[i]) if i < len(pose_vals)
                                         else 0.0 for i in range(len(gnames))]
                                _send_joint_pose(pub_topic, gvals, gnames)


# ─── 제어권 / 모델 상태 ───────────────────────────────────────────────────────

def render_status() -> dict:
    """제어권 + 모델 상태 placeholder 생성."""
    _start_listeners()
    ph_ctrl  = st.empty()
    ph_model = st.empty()
    return {"ctrl": ph_ctrl, "model": ph_model}


def update_status(placeholders: dict):
    """제어권 + 모델 상태 갱신."""
    if not placeholders:
        return

    # 제어권
    ctrl = _ctrl_state()
    with ctrl["lock"]:
        owner = ctrl["owner"]
    ph = placeholders["ctrl"]
    if owner is None:
        ph.info("제어권: 없음 (대기 중)")
    elif owner == CONTROL_SOURCE:
        ph.success(f"제어권: **{owner}** (이 웹)")
    else:
        ph.warning(f"제어권: **{owner}** (다른 클라이언트)")

    # 모델 상태
    ms = _model_state()
    with ms["lock"]:
        mstate = ms["state"]
        detail = ms["detail"]
    ph = placeholders["model"]
    STATE_DISPLAY = {
        "idle":    ("⚪", "모델 미로드"),
        "loading": ("🔄", "모델 로딩 중..."),
        "ready":   ("🟢", "모델 준비 완료"),
        "running": ("🤖", "추론 실행 중"),
        "error":   ("🔴", "모델 오류"),
    }
    icon, label = STATE_DISPLAY.get(mstate, ("❓", mstate))
    if mstate == "error" and detail:
        ph.error(f"{icon} {label}: {detail}")
    elif mstate == "running":
        ph.success(f"{icon} {label}")
    elif mstate == "ready":
        ph.success(f"{icon} {label}")
    elif mstate == "loading":
        ph.info(f"{icon} {label}")
    else:
        ph.info(f"{icon} {label}")


def wait_for_model_state(timeout: float = 1.5):
    """ModelServer의 첫 heartbeat을 받을 때까지 대기 (최초 페이지 로드용).

    heartbeat 주기는 1초이므로 timeout 1.5초면 첫 상태를 받기에 충분.
    """
    import time as _t
    _start_listeners()  # 리스너가 아직 시작 안 됐으면 시작
    ms = _model_state()
    deadline = _t.monotonic() + timeout
    while _t.monotonic() < deadline:
        with ms["lock"]:
            # heartbeat을 한 번이라도 받으면 _received 플래그가 생김
            if ms.get("_received"):
                return
        _t.sleep(0.05)


def get_model_state() -> str:
    """현재 모델 상태 문자열 반환."""
    ms = _model_state()
    with ms["lock"]:
        return ms["state"]
