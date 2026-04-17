"""pages/monitor.py — 실시간 상태 모니터링 + 포즈 제어"""

import time
import streamlit as st

from common.utils.ui_utils import (
    render_cameras,      update_cameras,
    render_robot,        update_robot,
    render_gripper,      update_gripper,
    render_pose_control,
    render_status,       update_status,
    get_model_state,
)


def show(cfg):
    st.header("Monitor")

    ph_cameras = render_cameras(cfg)
    st.divider()
    ph_robot   = render_robot(cfg)
    st.divider()
    ph_gripper = render_gripper(cfg)
    st.divider()

    # Pose 제어 — 추론 중이면 버튼 비활성화
    st.subheader("Control")
    inference_running = get_model_state() == "running"
    if inference_running:
        st.warning("추론 실행 중 — 수동 제어가 비활성화됩니다.")
    render_pose_control(cfg, disabled=inference_running)

    st.divider()
    st.subheader("Status")
    ph_status = render_status()

    while True:
        update_cameras(ph_cameras)
        update_robot(ph_robot, cfg)
        update_gripper(ph_gripper, cfg)
        update_status(ph_status)
        time.sleep(0.033)
