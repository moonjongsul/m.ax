import argparse
import signal
import time

from common.utils.utils import load_config
from common.vla_server import VlaServer
from common.comm_server import CommServer




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="gt_kitting")
    args = parser.parse_args()

    cfg = load_config(fname='config/server_config.yaml', project=args.project)

    comm = CommServer(cfg)
    comm.ros_comm_pipeline()

    # 메인 스레드: Ctrl+C까지 대기
    def _on_shutdown(sig, frame):
        print("\n[server] shutdown...")
        comm.shutdown()

    signal.signal(signal.SIGINT, _on_shutdown)
    signal.signal(signal.SIGTERM, _on_shutdown)

    while True:
        time.sleep(1)



if __name__ == "__main__":
    main()
