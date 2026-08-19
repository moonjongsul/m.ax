import multiprocessing as mp
import re
import threading
import time
from multiprocessing import shared_memory

import cv2
import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CompressedImage

IMG_TOPIC_NAME = "/observation/env/color/image_raw/compressed"

MODEL_ID = "google/gemma-4-E2B-it"
IMG_H, IMG_W = 480, 640
IMG_CHANNELS = 3
IMG_BUFFER_LEN = 6
IMG_BUFFER_DT = 0.333
MAX_NEW_TOKENS = 64

SHM_SLOTS = 12
SLOT_BYTES = IMG_H * IMG_W * IMG_CHANNELS

INSTRUCTION_HEADER = (
    f"다음은 {IMG_BUFFER_DT}초 간격으로 촬영된 {IMG_BUFFER_LEN}장의 시계열 이미지입니다. "
    "가장 오래된 프레임부터 가장 최근 프레임 순으로 제시됩니다."
)
INSTRUCTION_QUERY = (
    "주어지는 시계열 이미지를 보고 현재 로봇이 작업을 명령에 따라 잘 수행 중인지 다음의 형식에 맞춰서 설명해줘. "
    "** 같은 마크다운 기호는 사용하지 마. "
    "로봇에 주어진 작업 명령: flip object. "
    "success: true/false/processing "
    "robot status: moving / try to flip / grasping / hold 등 적절한 상태"
    "robot grasp: true/false "
    "robot location: "
)

_FIELD_KEYS = (
    "로봇에 주어진 작업 명령",
    "success",
    "robot status",
    "robot grasp",
    "robot location",
)


def _format_response(text: str) -> str:
    pattern = re.compile(
        r"\s*(?=(?:" + "|".join(re.escape(k) for k in _FIELD_KEYS) + r")\s*:)"
    )
    parts = [p.strip() for p in pattern.split(text) if p.strip()]
    return "\n".join(parts)


def inference_worker(shm_name: str, cmd_q: mp.Queue, result_q: mp.Queue):
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    shm = shared_memory.SharedMemory(name=shm_name)
    arr = np.ndarray(
        (SHM_SLOTS, IMG_H, IMG_W, IMG_CHANNELS), dtype=np.uint8, buffer=shm.buf
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()

    gen_cfg = model.generation_config
    gen_cfg.do_sample = False
    gen_cfg.num_beams = 1
    gen_cfg.use_cache = True
    if gen_cfg.pad_token_id is None:
        gen_cfg.pad_token_id = (
            processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
        )

    result_q.put({"type": "ready"})

    inf_count = 0
    while True:
        msg = cmd_q.get()
        if msg is None:
            break

        slot_indices = msg["slots"]
        req_id = msg["req_id"]

        frames = [arr[i].copy() for i in slot_indices]

        t0 = time.time()
        content = [{"type": "text", "text": INSTRUCTION_HEADER}]
        n = len(frames)
        for i, frame in enumerate(frames):
            t_ago = (n - 1 - i) * IMG_BUFFER_DT
            label = "현재 프레임" if i == n - 1 else f"{t_ago:.1f}초 전 프레임"
            content.append({"type": "image", "image": frame})
            content.append({"type": "text", "text": f"[프레임 {i + 1}/{n}] ({label})"})
        content.append({"type": "text", "text": INSTRUCTION_QUERY})

        messages = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.device)

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

        response = processor.decode(
            outputs[0][input_len:], skip_special_tokens=False
        )
        parsed = processor.parse_response(response).get("content")
        parsed = _format_response(parsed or "")

        inf_count += 1
        result_q.put({
            "type": "result",
            "req_id": req_id,
            "count": inf_count,
            "elapsed": time.time() - t0,
            "text": parsed,
        })

    shm.close()


class TaskManagerNode(Node):
    def __init__(self, shm_name: str, cmd_q: mp.Queue, result_q: mp.Queue):
        super().__init__("task_manager")
        self.cmd_q = cmd_q
        self.result_q = result_q

        self.shm = shared_memory.SharedMemory(name=shm_name)
        self.shm_arr = np.ndarray(
            (SHM_SLOTS, IMG_H, IMG_W, IMG_CHANNELS),
            dtype=np.uint8,
            buffer=self.shm.buf,
        )

        self.latest_msg = None
        self.latest_msg_lock = threading.Lock()
        self.buffer_lock = threading.Lock()
        self.write_idx = 0
        self.slot_history: list[int] = []
        self.inference_in_flight = False
        self.req_id = 0

        self._last_recv = None
        self._last_pub = None
        self._gaps_recv: list[float] = []
        self._gaps_pub: list[float] = []
        self._lags: list[float] = []
        self._stat_lock = threading.Lock()
        self.stat_timer = self.create_timer(2.0, self._stat_dump)

        self.io_group = MutuallyExclusiveCallbackGroup()
        self.buf_group = MutuallyExclusiveCallbackGroup()

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.img_sub = self.create_subscription(
            CompressedImage,
            IMG_TOPIC_NAME,
            self.img_callback,
            qos,
            callback_group=self.io_group,
        )
        self.buf_timer = self.create_timer(
            IMG_BUFFER_DT,
            self.buffer_callback,
            callback_group=self.buf_group,
        )

        self.result_thread = threading.Thread(
            target=self._result_loop, daemon=True
        )
        self.result_thread.start()

    def img_callback(self, msg: CompressedImage):
        with self.latest_msg_lock:
            self.latest_msg = msg
        now = time.time()
        pub_t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self._stat_lock:
            if self._last_recv is not None:
                self._gaps_recv.append(now - self._last_recv)
            if self._last_pub is not None:
                self._gaps_pub.append(pub_t - self._last_pub)
            self._lags.append(now - pub_t)
            self._last_recv = now
            self._last_pub = pub_t

    def _stat_dump(self):
        with self._stat_lock:
            gr = self._gaps_recv[:]
            gp = self._gaps_pub[:]
            la = self._lags[:]
            self._gaps_recv.clear()
            self._gaps_pub.clear()
            self._lags.clear()
        if not gr:
            return

        def s(name, arr):
            arr_s = sorted(arr)
            n = len(arr_s)
            return (
                f"{name}: n={n} "
                f"min={arr_s[0] * 1000:.1f}ms "
                f"p50={arr_s[n // 2] * 1000:.1f}ms "
                f"p95={arr_s[int(n * 0.95)] * 1000:.1f}ms "
                f"max={arr_s[-1] * 1000:.1f}ms"
            )

        self.get_logger().info(s("recv_gap", gr))
        self.get_logger().info(s("pub_gap ", gp))
        self.get_logger().info(s("lag     ", la))
        
    def buffer_callback(self):
        with self.latest_msg_lock:
            msg = self.latest_msg

        if msg is None:
            return

        buf = np.frombuffer(msg.data, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            return
        if bgr.shape[0] != IMG_H or bgr.shape[1] != IMG_W:
            bgr = cv2.resize(bgr, (IMG_W, IMG_H))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        with self.buffer_lock:
            slot = self.write_idx % SHM_SLOTS
            self.shm_arr[slot] = rgb
            self.write_idx += 1
            self.slot_history.append(slot)
            if len(self.slot_history) > IMG_BUFFER_LEN:
                self.slot_history.pop(0)

            ready = (
                len(self.slot_history) == IMG_BUFFER_LEN
                and not self.inference_in_flight
            )
            if ready:
                self.inference_in_flight = True
                self.req_id += 1
                req_id = self.req_id
                slots = list(self.slot_history)

        if ready:
            self.cmd_q.put({"req_id": req_id, "slots": slots})

        # self.get_logger().info(f"buffer")
        
    def _result_loop(self):
        while True:
            msg = self.result_q.get()
            if msg is None:
                return
            if msg.get("type") == "ready":
                self.get_logger().info("inference worker ready")
                continue
            if msg.get("type") != "result":
                continue

            with self.buffer_lock:
                self.inference_in_flight = False

            self.get_logger().info(
                f"============== [{msg['count']}] Total: {msg['elapsed']:.2f} sec ============="
            )
            self.get_logger().info(msg["text"])
            self.get_logger().info("=========================================")


def main(args=None):
    mp.set_start_method("spawn", force=True)

    shm = shared_memory.SharedMemory(
        create=True, size=SHM_SLOTS * SLOT_BYTES
    )
    cmd_q: mp.Queue = mp.Queue()
    result_q: mp.Queue = mp.Queue()

    worker = mp.Process(
        target=inference_worker,
        args=(shm.name, cmd_q, result_q),
        daemon=False,
    )
    worker.start()

    rclpy.init(args=args)
    node = TaskManagerNode(shm.name, cmd_q, result_q)
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        cmd_q.put(None)
        worker.join(timeout=5)
        if worker.is_alive():
            worker.terminate()
            worker.join()
        node.destroy_node()
        rclpy.shutdown()
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    main()
