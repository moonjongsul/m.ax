"""app_server.py — M.AX 통합 진입점

실행:
    python app_server.py [--project gt_kitting]

기동 순서:
    1. CommServer  — ROS ↔ ZMQ 브릿지 (robot/gripper/camera)
    2. ModelServer — VLA 모델 로드 + 추론 루프
    3. Streamlit   — 웹 UI (subprocess)
"""

import argparse
import signal
import subprocess
import sys
import time

from common.utils.utils import load_config
from common.comm_server import CommServer
from common.model_server import ModelServer


def main():
    parser = argparse.ArgumentParser(description="M.AX App Server")
    parser.add_argument("--project", type=str, default="gt_kitting")
    parser.add_argument("--no-web", action="store_true",
                        help="Streamlit 웹 UI 없이 서버만 실행")
    args = parser.parse_args()

    cfg = load_config(fname="config/server_config.yaml", project=args.project)

    # ── 1. CommServer 기동 ─────────────────────────────────────────────────
    comm = CommServer(cfg)
    comm.pipeline()

    # ── 2. ModelServer 기동 ────────────────────────────────────────────────
    # model = ModelServer(cfg)
    # model.start()

    # ── 3. Streamlit 웹 UI (subprocess) ────────────────────────────────────
    def _start_web():
        proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "web.py",
             "--server.headless", "true"],
        )
        print(f"[app_server] Streamlit started (pid={proc.pid})")
        print("[app_server] Web UI: http://localhost:8501")
        return proc

    web_proc = _start_web() if not args.no_web else None

    # ── Shutdown 핸들러 ────────────────────────────────────────────────────
    def _shutdown(_sig=None, _frame=None):
        print("\n[app_server] shutting down...")
        model.shutdown()
        comm.shutdown()
        if web_proc and web_proc.poll() is None:
            web_proc.terminate()
            web_proc.wait(timeout=5)
        print("[app_server] done.")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── 메인 루프 ──────────────────────────────────────────────────────────
    while True:
        if web_proc and web_proc.poll() is not None:
            rc = web_proc.returncode
            print(f"[app_server] Streamlit exited (rc={rc}). Restarting...")
            web_proc = _start_web()
        time.sleep(1)


if __name__ == "__main__":
    main()
