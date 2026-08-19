import time
import cv2
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# MODEL_ID = "google/gemma-4-E4B-it"
MODEL_ID = "google/gemma-4-E2B-it"
N_ITERS = 100
MAX_NEW_TOKENS = 64

img = cv2.imread("./kitting_target_img.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, [640, 480])
print(f"Image shape: {img.shape}")
pil_img = Image.fromarray(img)

processor = AutoProcessor.from_pretrained(MODEL_ID)

llm = LLM(
    model=MODEL_ID,
    dtype="bfloat16",
    trust_remote_code=True,
    limit_mm_per_prompt={"image": 4},
    max_model_len=4096,
)

sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=MAX_NEW_TOKENS,
)

INSTRUCTION_HEADER = (
    "다음은 1초 간격으로 촬영된 4장의 시계열 이미지입니다. "
    "가장 오래된 프레임부터 가장 최근 프레임 순으로 제시됩니다."
)
INSTRUCTION_QUERY = (
    "주어지는 시계열 이미지를 보고 현재 로봇이 작업을 명령에 따라 잘 수행 중인지 다음의 형식에 맞춰서 설명해줘. "
    "** 같은 마크다운 기호는 사용하지 마. "
    "로봇에 주어진 작업 명령: flip object. "
    "success: true/false "
    "robot status: moving / try to flip / grasping / hold 등 적절한 상태"
    "robot grasp: true/false "
    "object status: "
    "robot location: "
)


def build_prompt():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": INSTRUCTION_HEADER},
                {"type": "image"},
                {"type": "text", "text": "프레임 1/4 (3초 전 프레임)"},
                {"type": "image"},
                {"type": "text", "text": "프레임 2/4 (2초 전 프레임)"},
                {"type": "image"},
                {"type": "text", "text": "프레임 3/4 (1초 전 프레임)"},
                {"type": "image"},
                {"type": "text", "text": "프레임 4/4 (현재 프레임)"},
                {"type": "text", "text": INSTRUCTION_QUERY},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


prompt = build_prompt()
mm_data = {"image": [pil_img, pil_img, pil_img, pil_img]}

_ = llm.generate(
    [{"prompt": prompt, "multi_modal_data": mm_data}],
    sampling_params=SamplingParams(temperature=0.0, max_tokens=8),
)

for i in range(N_ITERS):
    t0 = time.time()
    request = {"prompt": prompt, "multi_modal_data": mm_data}
    outputs = llm.generate([request], sampling_params=sampling_params)
    end = time.time()

    out = outputs[0].outputs[0]
    response = out.text
    new_tokens = len(out.token_ids)
    print(response)
    print(f"time: {end - t0:.2f}s  ({new_tokens} tok, {new_tokens / (end - t0):.1f} tok/s)")
