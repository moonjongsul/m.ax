"""pages/monitor.py — 실시간 상태 모니터링 + 제어"""

import time
import streamlit as st

from common.utils.ui_utils import (
    render_cameras,       update_cameras,
    render_robot,         update_robot,
    render_gripper,       update_gripper,
    render_control,       update_control_status,
)


def show(cfg):
    ph_cameras = render_cameras(cfg)
    st.divider()
    ph_robot   = render_robot(cfg)
    st.divider()
    ph_gripper = render_gripper(cfg)
    st.divider()
    ph_ctrl    = render_control(cfg)

    while True:
        update_cameras(ph_cameras)
        update_robot(ph_robot, cfg)
        update_gripper(ph_gripper, cfg)
        update_control_status(ph_ctrl)
        time.sleep(0.033)
