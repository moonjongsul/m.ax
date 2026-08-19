import re
import threading
import time
from collections import deque

import cv2
import numpy as np
import rclpy
import torch
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CompressedImage
from transformers import AutoModelForMultimodalLM, AutoProcessor

IMG_TOPIC_NAME = "/observation/env/color/image_raw/compressed"

MODEL_ID = "google/gemma-4-E2B-it"
IMG_BUFFER_LEN = 6
IMG_BUFFER_DT = 0.333
MAX_NEW_TOKENS = 64



INSTRUCTION = (
    "이 이미지는 제조 도메인에서 로봇을 활용한 키팅 작업 수행을 위한 목표 이미지야. "
    "구체적으로 설명하자면, 트레이에 부품이 담겨진 모습이고, 트레이에는 보이는 것처럼 부품들이 담겨 있어. "
    "이 이미지를 보고 트레이에 담긴 부품들의 종류와 위치를 설명해줘. "
    "** 같은거 넣지 말고 응답은 아주 간략하게 해. "
)

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

class TaskRecognizerNode(Node):
    def __init__(self):
        super().__init__("task_recognizer")
        self.inf_count = 0
        self.latest_msg = None
        self.img_buffer = []
        
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        
        self.img_sub = self.create_subscription(
            CompressedImage,
            IMG_TOPIC_NAME,
            self.img_callback,
            10
        )
        self.viz_pub = self.create_publisher(CompressedImage, 
                                             "/task_manager/buffer_image/compressed",
                                             qos)
        
        self.buf_timer = self.create_timer(IMG_BUFFER_DT, self.buffer_callback)
        
        self.load_model()
        self.inf_timer = self.create_timer(1, self.inference_callback)
        
    def load_model(self):
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            )
        self.model.eval()
        
        gen_cfg = self.model.generation_config
        gen_cfg.do_sample = False
        gen_cfg.num_beams = 1
        gen_cfg.use_cache = True
        if gen_cfg.pad_token_id is None:
            gen_cfg.pad_token_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id

    def build_inputs(self, frames):
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
        
    def inference_callback(self):
        t0 = time.time()
        
        if len(self.img_buffer) < IMG_BUFFER_LEN:
            return
        
        frames = self.img_buffer.copy()
        
        inputs = self.build_inputs(frames)
        input_len = inputs['input_ids'].shape[-1]
        
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
            
        response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        parsed = self.processor.parse_response(response).get("content")
        parsed = self._format_response(parsed or "")
        
        elapsed = time.time() - t0
        
        self.inf_count += 1
        self.get_logger().info(f"============== [{self.inf_count}] Total: {elapsed:.2f} sec =============")
        self.get_logger().info(f"{parsed}")
        self.get_logger().info("=========================================")
        
    def img_callback(self, msg: CompressedImage):
        self.latest_msg = msg
        print('sub image')
        
    def buffer_callback(self):
        msg = self.latest_msg
        if msg is None:
            return
        
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.img_buffer.append(rgb)
        if len(self.img_buffer) > IMG_BUFFER_LEN:
            self.img_buffer.pop(0)
        
        self.get_logger().info(f"buffer: {len(self.img_buffer)}")
        
        # if len(self.img_buffer) == IMG_BUFFER_LEN:
        #     self.show_frames(self.img_buffer)
    
    @staticmethod
    def _format_response(text: str) -> str:
        pattern = re.compile(r"\s*(?=(?:" + "|".join(re.escape(k) for k in _FIELD_KEYS) + r")\s*:)")
        parts = [p.strip() for p in pattern.split(text) if p.strip()]
        return "\n".join(parts)
    
    def show_frames(self, frames):
        n = len(frames)
        bgr_frames1 = []
        bgr_frames2 = []
        for i, f in enumerate(frames):
            bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR).copy()
            t_ago = (n - 1 - i) * IMG_BUFFER_DT
            label = "current" if i == n - 1 else f"-{t_ago:.1f}s"
            cv2.putText(
                bgr, f"[{i + 1}/{n}] {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )
            if i < len(frames) / 2:
                bgr_frames1.append(bgr)
            else:
                bgr_frames2.append(bgr)
        canvas1 = np.hstack(bgr_frames1)
        canvas2 = np.hstack(bgr_frames2)
        canvas = np.vstack([canvas1, canvas2])
        
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
        _, encoded = cv2.imencode('.jpg', canvas, encode_params)
        
        if _:
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera_color_optical_frame'
            msg.format = 'jpeg'  # 'jpeg' 또는 'png'
            msg.data = encoded.tobytes()
            
            self.viz_pub.publish(msg)
            
def main(args=None):
    rclpy.init(args=args)
    node = TaskRecognizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()