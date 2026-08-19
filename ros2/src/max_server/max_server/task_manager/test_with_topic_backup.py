import re
import threading
import time
from collections import deque

import cv2
import numpy as np
import rclpy
import torch
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CompressedImage, Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

# MODEL_ID = "google/gemma-4-E4B-it"
MODEL_ID = "google/gemma-4-E2B-it"
TOPIC_NAME = "/observation/env/color/image_raw/compressed"
IMG_BUFFER_LEN = 5
IMG_BUFFER_DT = 0.2
MAX_NEW_TOKENS = 64

INSTRUCTION_HEADER = (
    f"다음은 {IMG_BUFFER_DT}초 간격으로 촬영된 {IMG_BUFFER_LEN}장의 시계열 이미지입니다. "
    "가장 오래된 프레임부터 가장 최근 프레임 순으로 제시됩니다."
)
INSTRUCTION_QUERY = (
    "주어지는 시계열 이미지를 보고 현재 로봇이 작업을 명령에 따라 잘 수행 중인지 다음의 형식에 맞춰서 설명해줘. "
    "로봇에 장착된 그리퍼를 유의깊게 보고 판단해줘. "
    "** 같은 마크다운 기호는 사용하지 마. "
    "로봇에 주어진 작업 명령: flip object. "
    "success: true/false "
    "robot status: moving / try to flip / grasping / hold 등 적절한 상태"
    "robot grasp: true/false "
    "robot location: on green "
)


_FIELD_KEYS = (
    "로봇에 주어진 작업 명령",
    "success",
    "robot status",
    "robot grasp",
    "object status",
    "robot location",
)


def _format_response(text: str) -> str:
    pattern = re.compile(r"\s*(?=(?:" + "|".join(re.escape(k) for k in _FIELD_KEYS) + r")\s*:)")
    parts = [p.strip() for p in pattern.split(text) if p.strip()]
    return "\n".join(parts)


class TaskRecognizerNode(Node):
    def __init__(self):
        super().__init__("task_recognizer")

        self.buffer_lock = threading.Lock()
        self.latest_msg = None
        self.latest_msg_lock = threading.Lock()
        self.img_buffer: deque = deque(maxlen=IMG_BUFFER_LEN)

        sub_group = MutuallyExclusiveCallbackGroup()
        timer_group = MutuallyExclusiveCallbackGroup()

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.sub = self.create_subscription(
            CompressedImage, TOPIC_NAME, self._on_image, qos, callback_group=sub_group
        )
        self.viz_pub = self.create_publisher(Image, "/task_manager/model_input", 1)
        self.buffer_timer = self.create_timer(
            IMG_BUFFER_DT, self._on_buffer_tick, callback_group=timer_group
        )

        self.get_logger().info(
            f"Subscribed: {TOPIC_NAME} | buffer dt={IMG_BUFFER_DT}s len={IMG_BUFFER_LEN}"
        )

        self._load_model()

        self.stop_event = threading.Event()

    def _load_model(self):
        self.get_logger().info(f"Loading model: {MODEL_ID}")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="sdpa",
        )
        self.model.eval()
        hf_map = getattr(self.model, "hf_device_map", None)
        param_devs = {str(p.device) for p in self.model.parameters()}
        self.get_logger().info(f"hf_device_map: {hf_map} | param devices: {param_devs}")

        gen_cfg = self.model.generation_config
        gen_cfg.do_sample = False
        gen_cfg.num_beams = 1
        gen_cfg.use_cache = True
        if gen_cfg.pad_token_id is None:
            gen_cfg.pad_token_id = (
                self.processor.tokenizer.pad_token_id
                or self.processor.tokenizer.eos_token_id
            )

        self.get_logger().info("Warming up model...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        warm_frames = [dummy] * IMG_BUFFER_LEN
        warm_inputs = self._build_inputs(warm_frames)
        with torch.inference_mode():
            _ = self.model.generate(**warm_inputs, max_new_tokens=8)
        torch.cuda.synchronize()
        self.get_logger().info("Model ready.")

        # Pre-spin baseline: ROS executor가 돌기 전에 동일 generate 한 번 시간 측정
        self.get_logger().info("Pre-spin generate benchmark...")
        bench_inputs = self._build_inputs(warm_frames)
        bench_input_len = bench_inputs["input_ids"].shape[-1]
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            bench_out = self.model.generate(
                **bench_inputs, max_new_tokens=MAX_NEW_TOKENS
            )
        torch.cuda.synchronize()
        bench_elapsed = time.time() - t0
        bench_new = bench_out.shape[-1] - bench_input_len
        self.get_logger().info(
            f"[BASELINE] gen: {bench_elapsed:.2f}s  prompt_tok={bench_input_len}  "
            f"new_tok={bench_new}  ({bench_new / bench_elapsed:.1f} tok/s)"
        )

    def _on_image(self, msg: CompressedImage):
        with self.latest_msg_lock:
            self.latest_msg = msg

    def _on_buffer_tick(self):
        with self.latest_msg_lock:
            msg = self.latest_msg
        if msg is None:
            return
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with self.buffer_lock:
            self.img_buffer.append(rgb)

    def _snapshot_buffer(self):
        with self.buffer_lock:
            if len(self.img_buffer) < IMG_BUFFER_LEN:
                return None
            return list(self.img_buffer)

    def _build_inputs(self, frames):
        content = [{"type": "text", "text": INSTRUCTION_HEADER}]
        n = len(frames)
        for i, frame in enumerate(frames):
            t_ago = (n - 1 - i) * IMG_BUFFER_DT
            label = "현재 프레임" if i == n - 1 else f"{t_ago:.1f}초 전 프레임"
            content.append({"type": "image", "image": frame})
            content.append({"type": "text", "text": f"[프레임 {i + 1}/{n}] ({label})"})
        content.append({"type": "text", "text": INSTRUCTION_QUERY})

        messages = [{"role": "user", "content": content}]
        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device)

    def _show_frames(self, frames):
        n = len(frames)
        bgr_frames = []
        for i, f in enumerate(frames):
            bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR).copy()
            t_ago = (n - 1 - i) * IMG_BUFFER_DT
            label = "current" if i == n - 1 else f"-{t_ago:.1f}s"
            cv2.putText(
                bgr, f"[{i + 1}/{n}] {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )
            bgr_frames.append(bgr)
        canvas = np.hstack(bgr_frames)

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "model_input"
        msg.height, msg.width = canvas.shape[:2]
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = canvas.shape[1] * 3
        msg.data = canvas.tobytes()
        self.viz_pub.publish(msg)

    def run_once(self):
        frames = self._snapshot_buffer()
        if frames is None:
            return

        self._show_frames(frames)

        t0 = time.time()
        inputs = self._build_inputs(frames)
        torch.cuda.synchronize()
        t_build = time.time()
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        torch.cuda.synchronize()
        t_gen = time.time()

        response = self.processor.decode(
            outputs[0][input_len:], skip_special_tokens=False
        )
        parsed = self.processor.parse_response(response)
        elapsed = time.time() - t0
        new_tokens = outputs.shape[-1] - input_len

        print("=" * 60)
        print(_format_response(parsed.get("content") or ""))
        print(
            f"total: {elapsed:.2f}s  build: {t_build - t0:.2f}s  "
            f"gen: {t_gen - t_build:.2f}s  decode: {elapsed - (t_gen - t0):.2f}s  "
            f"prompt_tok={input_len}  new_tok={new_tokens}  "
            f"({new_tokens / (t_gen - t_build):.1f} tok/s)"
        )

    def shutdown(self):
        self.stop_event.set()


def main():
    rclpy.init()
    node = TaskRecognizerNode()
    try:
        while rclpy.ok() and not node.stop_event.is_set():
            rclpy.spin_once(node, timeout_sec=0.0)
            node.run_once()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
