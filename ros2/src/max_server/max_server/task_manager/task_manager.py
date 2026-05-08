import gc
import torch
from collections import deque
from transformers import AutoProcessor, AutoModelForMultimodalLM


class TaskManager:

    def __init__(
        self,
        node,
        model: str = "",
        instruction: str = "",
        observation_list: list[str] | None = None,
        image_num: int = 0,
        image_dt: float = 0.0,
    ):
        self._node = node
        self.model_name = model
        self.instruction = instruction
        self.observation_list = list(observation_list or [])
        
        self.image_num = image_num
        self.image_dt = image_dt
        self.image_buffer = deque(maxlen=self.image_num)
        
        self.processor = None
        self.model = None
        self.fps = 0.0
        
        self.load()

    def load(self):
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForMultimodalLM.from_pretrained(
                self.model_name,
                dtype='auto',
                device_map='auto'
                )
        except Exception as e:
            self._node.get_logger().error(f"Failed to load TaskManager model: {e}")
            self.processor = None
            self.model = None
            
    def unload(self):
        had_model = self.model is not None

        # accelerate가 device_map='auto'로 올린 모델은 .to("cpu")가 실패할 수 있음.
        if self.model is not None:
            try:
                self.model.to("cpu")
            except Exception:
                pass

        self.model = None
        self.processor = None

        if had_model:
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass
        
    def predict(
        self,
        images,
        instruction,
        
    ) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": images},
                    {"type": "text", "text": instruction}
                ]
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device)
        
        input_len = inputs["input_ids"].shape[-1]

        # Generate output
        outputs = self.model.generate(**inputs, max_new_tokens=512)
        response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=False)

        # Parse output
        output = self.processor.parse_response(response)

        return output
        
    def update_image_buffer(self, image):
        if self.image_num:
            self.image_buffer.append(image)
            if len(self.image_buffer) > self.image_num:
                self.image_buffer.pop(0) 
                
    
        