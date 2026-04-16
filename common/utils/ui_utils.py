"""common/utils/ui_utils.py

Streamlit UI 컴포넌트 모음.
각 render_*() 함수는 placeholder dict를 반환하고,
update_*() 함수는 해당 placeholder에 최신 데이터를 밀어 넣는다.

사용 예 (pages/monitor.py):
    ph_cameras = render_cameras(cfg)
    ph_robot   = render_robot(cfg)
    ph_gripper = render_gripper(cfg)
    ph_ctrl    = render_control(cfg)

    while True:
        update_cameras(ph_cameras)
        update_robot(ph_robot, cfg)
        update_gripper(ph_gripper, cfg)
        update_control_status(ph_ctrl)
        time.sleep(0.033)
"""

import base64
import pickle
import threading

import cv2
import numpy as np
import streamlit as st
import zmq
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

# ─── ZMQ 설정 ────────────────────────────────────────────────────────────────

ZMQ_HOST       = "localhost"
ZMQ_BASE_PUB   = 5570
ZMQ_PULL_PORT  = 5590   # app → CommServer Arbiter
ZMQ_CTRL_PORT  = 5591   # CommServer → app (제어권 알림)

CAMERA_DOMAIN  = 1
ROBOT_DOMAIN   = 0

CONTROL_SOURCE = "web"   # 이 클라이언트의 식별자

# ─── 공통 헬퍼 ───────────────────────────────────────────────────────────────

ROTATE_MAP = {
    90:  cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def _to_html_img(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf).decode()
    return f'<img src="data:image/jpeg;base64,{b64}" style="width:100%;border-radius:4px;">'


def _no_signal_html() -> str:
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(img, "No Signal", (55, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (160, 160, 160), 2)
    return _to_html_img(img)


@st.cache_resource
def _zmq_context() -> zmq.Context:
    """프로세스 전체에서 단일 ZMQ Context 공유."""
    return zmq.Context()


def _new_sub_socket(port: int, topics: tuple[str, ...]) -> zmq.Socket:
    ctx = _zmq_context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVTIMEO, 100)
    sock.setsockopt(zmq.RCVHWM, 1)
    for t in topics:
        sock.setsockopt_string(zmq.SUBSCRIBE, t)
    sock.connect(f"tcp://{ZMQ_HOST}:{port}")
    return sock


@st.cache_resource
def _camera_sub_socket(port: int, topic: str) -> zmq.Socket:
    return _new_sub_socket(port, (topic,))


@st.cache_resource
def _robot_sub_socket(port: int, topics: tuple[str, ...]) -> zmq.Socket:
    return _new_sub_socket(port, topics)


@st.cache_resource
def _gripper_sub_socket(port: int, topics: tuple[str, ...]) -> zmq.Socket:
    return _new_sub_socket(port, topics)


def _recv_latest(sock: zmq.Socket) -> tuple[str, bytes] | None:
    result = None
    try:
        while True:
            parts = sock.recv_multipart(zmq.NOBLOCK)
            if len(parts) == 2:
                try:
                    topic = parts[0].decode("utf-8")
                    result = (topic, parts[1])
                except UnicodeDecodeError:
                    pass   # 잘못된 프레임 무시
    except zmq.Again:
        pass
    return result


def _compressed_image_to_bgr(raw: bytes, rotate: int = 0) -> np.ndarray | None:
    msg = pickle.loads(raw)
    arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    rot = ROTATE_MAP.get(rotate)
    if rot is not None:
        img = cv2.rotate(img, rot)
    return img


def _joint_state_to_dict(raw: bytes) -> dict[str, float]:
    msg = pickle.loads(raw)
    return {name: round(float(pos), 4)
            for name, pos in zip(msg.name, msg.position)}


def _float32_to_value(raw: bytes) -> float:
    msg = pickle.loads(raw)
    return round(float(msg.data), 4)


# ─── ZMQ 제어 소켓 (app → Arbiter) ──────────────────────────────────────────

@st.cache_resource
def _control_push_socket() -> zmq.Socket:
    """CommServer Arbiter로 제어 명령을 보내는 PUSH 소켓."""
    ctx = _zmq_context()
    sock = ctx.socket(zmq.PUSH)
    sock.setsockopt(zmq.SNDHWM, 10)
    sock.connect(f"tcp://{ZMQ_HOST}:{ZMQ_PULL_PORT}")
    return sock


@st.cache_resource
def _ctrl_state() -> dict:
    """제어권 상태를 공유하는 thread-safe dict."""
    return {"owner": None, "lock": threading.Lock()}


@st.cache_resource
def _ctrl_listener_started() -> dict:
    """백그라운드 ctrl listener가 한 번만 시작되도록 보장하는 sentinel."""
    ctx = _zmq_context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVTIMEO, 300)
    sock.setsockopt_string(zmq.SUBSCRIBE, "control")
    sock.connect(f"tcp://{ZMQ_HOST}:{ZMQ_CTRL_PORT}")

    def _listen():
        state = _ctrl_state()
        while True:
            try:
                _, msg = sock.recv_multipart()
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

    t = threading.Thread(target=_listen, daemon=True, name="ctrl-listener")
    t.start()
    return {"started": True}


def _start_ctrl_listener() -> None:
    _ctrl_listener_started()  # cache_resource로 한 번만 실행됨


def _send_control(topic: str, raw_msg: bytes):
    """[source, topic, pickled_msg] → Arbiter PUSH."""
    sock = _control_push_socket()
    sock.send_multipart([
        CONTROL_SOURCE.encode(),
        topic.encode(),
        raw_msg,
    ])


def _make_joint_state_msg(positions: list[float], names: list[str]) -> bytes:
    msg = JointState()
    msg.header = Header()
    msg.name = names
    msg.position = positions
    msg.velocity = [0.0] * len(positions)
    msg.effort   = [0.0] * len(positions)
    return pickle.dumps(msg)


def _make_float32_msg(value: float) -> bytes:
    from std_msgs.msg import Float32
    msg = Float32()
    msg.data = value
    return pickle.dumps(msg)


# ─── Camera ──────────────────────────────────────────────────────────────────

@st.cache_resource
def _camera_sockets(topic_rotate_pairs: tuple[tuple[str, int], ...]) -> dict[str, tuple[zmq.Socket, int]]:
    port = ZMQ_BASE_PUB + CAMERA_DOMAIN
    return {
        topic: (_camera_sub_socket(port, topic), rotate)
        for topic, rotate in topic_rotate_pairs
    }


def render_cameras(cfg) -> dict:
    cameras: dict = cfg.env.get("camera") or {}
    cam_list = []
    for _, cam_cfg in cameras.items():
        topics = list((cam_cfg.topic.get("subscribe") or {}).keys())
        if not topics:
            continue
        cam_list.append({
            "name":   cam_cfg.name,
            "topic":  topics[0],
            "rotate": int(cam_cfg.get("rotate", 0)),
        })

    if not cam_list:
        return {}

    st.subheader("Cameras")
    n_cols = min(len(cam_list), 4)
    cols = st.columns(n_cols)

    placeholders: dict[str, dict] = {}
    for i, cam in enumerate(cam_list):
        with cols[i % n_cols]:
            st.markdown(f"**{cam['name']}**")
            st.caption(cam["topic"])
            placeholders[cam["topic"]] = {
                "placeholder": st.empty(),
                "rotate":      cam["rotate"],
                "last_frame":  None,
            }

    return placeholders


def update_cameras(placeholders: dict):
    if not placeholders:
        return

    topic_rotate = tuple((t, v["rotate"]) for t, v in placeholders.items())
    sockets = _camera_sockets(topic_rotate)

    for topic, meta in placeholders.items():
        sock, rotate = sockets[topic]
        result = _recv_latest(sock)
        if result is not None:
            _, raw = result
            frame = _compressed_image_to_bgr(raw, rotate)
            if frame is not None:
                meta["last_frame"] = frame

        frame = meta["last_frame"]
        html = _to_html_img(frame) if frame is not None else _no_signal_html()
        meta["placeholder"].html(html)


# ─── Robot ───────────────────────────────────────────────────────────────────

@st.cache_resource
def _robot_sockets(sub_topics: tuple[str, ...]) -> zmq.Socket:
    port = ZMQ_BASE_PUB + ROBOT_DOMAIN
    return _robot_sub_socket(port, sub_topics)


def render_robot(cfg) -> dict:
    robot = (cfg.env or {}).get("robot")
    if not robot:
        return {}

    sub_topics = list((robot.topic.get("subscribe") or {}).keys())
    pub_topics  = list((robot.topic.get("publish")   or {}).keys())

    st.subheader("Robot")
    col_sub, col_pub = st.columns(2)

    placeholders: dict[str, dict] = {}

    with col_sub:
        st.markdown("**Subscribe** (ROS → ZMQ)")
        for t in sub_topics:
            st.caption(t)
            ph = st.empty()
            ph.caption("⏳ 토픽 대기 중...")
            placeholders[f"sub:{t}"] = {"ph": ph, "received": False}

    with col_pub:
        st.markdown("**Publish** (ZMQ → ROS)")
        for t in pub_topics:
            st.caption(t)
            ph = st.empty()
            ph.caption("⏳ 토픽 대기 중...")
            placeholders[f"pub:{t}"] = {"ph": ph, "received": False}

    return placeholders


def update_robot(placeholders: dict, cfg):
    if not placeholders:
        return

    robot = (cfg.env or {}).get("robot")
    if not robot:
        return

    sub_topics = tuple((robot.topic.get("subscribe") or {}).keys())
    sock = _robot_sockets(sub_topics)
    result = _recv_latest(sock)
    if result is None:
        return

    topic, raw = result
    key = f"sub:{topic}"
    if key in placeholders:
        joints = _joint_state_to_dict(raw)
        rows = "\n".join(f"- `{n}`: {v}" for n, v in joints.items())
        placeholders[key]["ph"].markdown(rows)
        placeholders[key]["received"] = True
        # 제어 UI 초기값으로 사용하기 위해 최신 joint 상태 저장
        st.session_state["_robot_current_joints"] = joints


# ─── Gripper ─────────────────────────────────────────────────────────────────

@st.cache_resource
def _gripper_sockets(sub_topics: tuple[str, ...]) -> zmq.Socket:
    port = ZMQ_BASE_PUB + ROBOT_DOMAIN
    return _gripper_sub_socket(port, sub_topics)


def render_gripper(cfg) -> dict:
    gripper = (cfg.env or {}).get("gripper")
    if not gripper:
        return {}

    sub_topics = list((gripper.topic.get("subscribe") or {}).keys())
    pub_topics  = list((gripper.topic.get("publish")   or {}).keys())

    st.subheader("Gripper")
    col_sub, col_pub = st.columns(2)

    placeholders: dict[str, dict] = {}

    with col_sub:
        st.markdown("**Subscribe** (ROS → ZMQ)")
        for t in sub_topics:
            st.caption(t)
            ph = st.empty()
            ph.caption("⏳ 토픽 대기 중...")
            placeholders[f"sub:{t}"] = {"ph": ph, "received": False}

    with col_pub:
        st.markdown("**Publish** (ZMQ → ROS)")
        for t in pub_topics:
            st.caption(t)
            ph = st.empty()
            ph.caption("⏳ 토픽 대기 중...")
            placeholders[f"pub:{t}"] = {"ph": ph, "received": False}

    return placeholders


def update_gripper(placeholders: dict, cfg):
    if not placeholders:
        return

    gripper = (cfg.env or {}).get("gripper")
    if not gripper:
        return

    sub_topics = tuple((gripper.topic.get("subscribe") or {}).keys())
    sock = _gripper_sockets(sub_topics)
    result = _recv_latest(sock)
    if result is None:
        return

    topic, raw = result
    key = f"sub:{topic}"
    if key in placeholders:
        joints = _joint_state_to_dict(raw)
        rows = "\n".join(f"- `{n}`: {v}" for n, v in joints.items())
        placeholders[key]["ph"].markdown(rows)
        placeholders[key]["received"] = True
        st.session_state["_gripper_current_joints"] = joints


# ─── Control ─────────────────────────────────────────────────────────────────

def render_control(cfg) -> dict:
    """제어권 상태 + 로봇/그리퍼 제어 UI 렌더링.

    반환값:
        {"status_ph": st.empty}
    """
    _start_ctrl_listener()

    robot   = (cfg.env or {}).get("robot")
    gripper = (cfg.env or {}).get("gripper")

    st.subheader("Control")
    status_ph = st.empty()
    st.divider()

    # ── 로봇 조인트 제어 ──────────────────────────────────────────────────
    if robot:
        pub_topics = list((robot.topic.get("publish") or {}).keys())
        poses: dict = dict(robot.get("pose") or {})   # yaml pose 목록

        for topic in pub_topics:
            st.markdown(f"**Robot** `{topic}`")

            # 현재 수신 중인 joint 값을 슬라이더 초기값으로 사용 (최초 1회만)
            current: dict[str, float] = st.session_state.get(
                "_robot_current_joints", {}
            )
            joint_names = list(current.keys()) if current else \
                          [f"joint{i+1}" for i in range(7)]

            # session_state에 아직 없는 joint만 현재값으로 초기화
            for name, val in zip(joint_names, [current.get(n, 0.0) for n in joint_names]):
                skey = f"ctrl_robot_{topic}_{name}"
                if skey not in st.session_state:
                    st.session_state[skey] = val

            # pose 버튼으로 슬라이더 값 덮어쓰기
            if poses:
                pose_cols = st.columns(len(poses))
                for col, (pose_name, pose_vals) in zip(pose_cols, poses.items()):
                    with col:
                        if st.button(
                            pose_name.capitalize(),
                            key=f"pose_robot_{topic}_{pose_name}",
                        ):
                            for i, n in enumerate(joint_names):
                                st.session_state[f"ctrl_robot_{topic}_{n}"] = \
                                    float(pose_vals[i]) if i < len(pose_vals) else 0.0

            # 슬라이더 — value= 생략, session_state 키로만 관리
            cols = st.columns(len(joint_names))
            positions = []
            for col, name in zip(cols, joint_names):
                with col:
                    val = st.slider(
                        name,
                        min_value=-3.14, max_value=3.14,
                        step=0.01,
                        key=f"ctrl_robot_{topic}_{name}",
                    )
                    positions.append(val)

            if st.button("Send", key=f"send_robot_{topic}"):
                raw = _make_joint_state_msg(positions, joint_names)
                _send_control(topic, raw)

    st.divider()

    # ── 그리퍼 제어 ──────────────────────────────────────────────────────
    if gripper:
        pub_topics  = list((gripper.topic.get("publish") or {}).keys())
        pub_infos   = list((gripper.topic.get("publish") or {}).values())

        poses: dict = dict(gripper.get("pose") or {})

        for topic, info in zip(pub_topics, pub_infos):
            st.markdown(f"**Gripper** `{topic}`")
            if "Float32" in info.type:
                # pose 버튼 (Float32: pose 값의 첫 번째 요소를 0~100% 범위로 사용)
                skey = f"ctrl_gripper_{topic}"
                if skey not in st.session_state:
                    st.session_state[skey] = 50.0

                if poses:
                    pose_cols = st.columns(len(poses))
                    for col, (pose_name, pose_vals) in zip(pose_cols, poses.items()):
                        with col:
                            if st.button(
                                pose_name.capitalize(),
                                key=f"pose_gripper_{topic}_{pose_name}",
                            ):
                                st.session_state[skey] = float(pose_vals[0]) * 100.0

                val = st.slider(
                    "width (%)",
                    min_value=0.0, max_value=100.0,
                    step=1.0,
                    key=skey,
                )
                if st.button("Send", key=f"send_gripper_{topic}"):
                    raw = _make_float32_msg(val)
                    _send_control(topic, raw)
            else:
                current_g: dict[str, float] = st.session_state.get(
                    "_gripper_current_joints", {}
                )
                gnames = list(current_g.keys()) if current_g else ["gripper_joint"]
                for name in gnames:
                    gskey = f"ctrl_gripper_{topic}_{name}"
                    if gskey not in st.session_state:
                        st.session_state[gskey] = current_g.get(name, 0.0)
                gcols = st.columns(len(gnames))
                gvals = []
                for col, name in zip(gcols, gnames):
                    with col:
                        v = st.slider(
                            name,
                            min_value=-1.57, max_value=1.57,
                            step=0.01,
                            key=f"ctrl_gripper_{topic}_{name}",
                        )
                        gvals.append(v)
                if st.button("Send", key=f"send_gripper_{topic}"):
                    raw = _make_joint_state_msg(gvals, gnames)
                    _send_control(topic, raw)

    return {"status_ph": status_ph}


def update_control_status(placeholders: dict):
    """제어권 현황을 status placeholder에 갱신."""
    if not placeholders:
        return

    state = _ctrl_state()
    with state["lock"]:
        owner = state["owner"]

    ph = placeholders["status_ph"]
    if owner is None:
        ph.info("제어권: 없음 (대기 중)")
    elif owner == CONTROL_SOURCE:
        ph.success(f"제어권: **{owner}** (현재 이 웹)")
    else:
        ph.warning(f"제어권: **{owner}** (다른 클라이언트 점유 중)")
