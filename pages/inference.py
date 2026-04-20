"""pages/inference.py — VLA 추론 실행"""

import time
import streamlit as st

from common.utils.ui_utils import (
    render_cameras,      update_cameras,
    render_robot,        update_robot,
    render_gripper,      update_gripper,
    render_pose_control,
    render_status,       update_status,
    get_model_state,     wait_for_model_state,
    send_model_start,    send_model_stop,
    send_model_reload,
)

_PROMPT_KEY    = "inference_prompt"
_CKPT_KEY      = "inference_checkpoint"


def _init_session(cfg):
    """session_state 초기값 설정 (최초 1회)."""
    model_cfg = cfg.get("model") or {}
    if _PROMPT_KEY not in st.session_state:
        st.session_state[_PROMPT_KEY] = str(model_cfg.get("task_prompt") or "")
    if _CKPT_KEY not in st.session_state:
        st.session_state[_CKPT_KEY] = str(model_cfg.get("checkpoint") or "")


def _render_model_panel(cfg):
    """모델 설정 + 추론 시작/정지 패널 렌더링. 상태 placeholder 반환."""
    model_cfg = cfg.get("model") or {}
    mstate    = get_model_state()

    st.subheader("Model")

    # ── 정적 정보 ────────────────────────────────────────────────────────
    col_info, col_ctrl = st.columns([2, 1])
    with col_info:
        policy = model_cfg.get("policy", "—")
        device = model_cfg.get("device", "—")
        dtype  = model_cfg.get("dtype", "—")
        st.markdown(
            f"**Policy:** `{policy}` &nbsp;|&nbsp; "
            f"**Device:** `{device}` &nbsp;|&nbsp; "
            f"**Dtype:** `{dtype}`"
        )

    # ── Checkpoint 입력 ───────────────────────────────────────────────────
    st.text_input(
        "Checkpoint 경로",
        key=_CKPT_KEY,
        placeholder="/path/to/checkpoint 또는 HuggingFace repo ID",
        help="변경 후 '모델 재로드' 버튼을 눌러주세요.",
    )

    col_reload, _ = st.columns([1, 3])
    with col_reload:
        reload_disabled = mstate in ("loading", "running")
        if st.button("모델 재로드", disabled=reload_disabled, use_container_width=True):
            ckpt = st.session_state[_CKPT_KEY].strip()
            if ckpt:
                send_model_reload(ckpt)
            else:
                st.warning("Checkpoint 경로를 입력해주세요.")

    st.divider()

    # ── 프롬프트 입력 ────────────────────────────────────────────────────
    st.text_area(
        "Task Prompt",
        key=_PROMPT_KEY,
        height=80,
        help="추론에 사용할 언어 지시문. 변경 후 시작 버튼을 누르면 적용됩니다.",
    )

    # ── 시작 / 정지 버튼 ─────────────────────────────────────────────────
    col_start, col_stop = st.columns(2)
    with col_start:
        start_disabled = mstate != "ready"
        if st.button("▶ 추론 시작", disabled=start_disabled,
                     use_container_width=True, type="primary"):
            prompt = st.session_state[_PROMPT_KEY].strip()
            send_model_start(prompt)

    with col_stop:
        stop_disabled = mstate != "running"
        if st.button("■ 추론 정지", disabled=stop_disabled,
                     use_container_width=True):
            send_model_stop()

    if mstate == "idle":
        st.info("모델이 로드되지 않았습니다. Checkpoint를 입력하고 재로드해주세요.")
    elif mstate == "error":
        st.error("모델 로드에 실패했습니다. Checkpoint 경로를 확인해주세요.")

    # ── 상태 placeholder ─────────────────────────────────────────────────
    st.divider()
    st.subheader("Status")
    return render_status()


def show(cfg):
    st.header("Inference")
    _init_session(cfg)

    # ModelServer heartbeat 수신 대기 (첫 페이지 로드 시 버튼 상태 정확히 반영)
    wait_for_model_state()

    # ── 카메라 + 로봇/그리퍼 상태 ────────────────────────────────────────
    ph_cameras = render_cameras(cfg)
    st.divider()
    ph_robot   = render_robot(cfg)
    st.divider()
    ph_gripper = render_gripper(cfg)
    st.divider()

    # ── Pose 제어 (추론 중 비활성화) ──────────────────────────────────────
    st.subheader("Control")
    inference_running = get_model_state() == "running"
    if inference_running:
        st.warning("추론 실행 중 — 수동 제어가 비활성화됩니다.")
    render_pose_control(cfg, disabled=inference_running)
    st.divider()

    # ── 모델 패널 ─────────────────────────────────────────────────────────
    ph_status = _render_model_panel(cfg)

    # ── 실시간 업데이트 루프 ──────────────────────────────────────────────
    while True:
        update_cameras(ph_cameras)
        update_robot(ph_robot, cfg)
        update_gripper(ph_gripper, cfg)
        update_status(ph_status)
        time.sleep(0.033)
