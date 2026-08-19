import time
import cv2
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

# MODEL_ID = "google/gemma-4-E4B-it"
MODEL_ID = "google/gemma-4-E2B-it"
N_ITERS = 100
MAX_NEW_TOKENS = 64

img = cv2.imread("./kitting_target_img.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, [640, 480])
print(f"Image shape: {img.shape}")

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
gen_cfg.cache_implementation = "static"
if gen_cfg.pad_token_id is None:
    gen_cfg.pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

lm = getattr(model, "language_model", None) or getattr(model, "model", None)
if lm is not None and hasattr(lm, "forward"):
    lm.forward = torch.compile(
        lm.forward,
        mode="reduce-overhead",
        fullgraph=False,
        dynamic=False,
    )
    print(f"Compiled language model: {type(lm).__name__}")
else:
    model.forward = torch.compile(
        model.forward,
        mode="reduce-overhead",
        fullgraph=False,
        dynamic=False,
    )
    print("Compiled top-level model.forward")

instruction = (
    "이 이미지는 제조 도메인에서 로봇을 활용한 키팅 작업 수행을 위한 목표 이미지야. "
    "구체적으로 설명하자면, 트레이에 부품이 담겨진 모습이고, 트레이에는 보이는 것처럼 부품들이 담겨 있어. "
    "이 이미지를 보고 트레이에 담긴 부품들의 종류와 위치를 설명해줘. "
    "** 같은거 넣지 말고 응답은 아주 간략하게 해. "
)

INSTRUCTION_HEADER = (
    f"다음은 1초 간격으로 촬영된 4장의 시계열 이미지입니다. "
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



def build_inputs(image, text):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": INSTRUCTION_HEADER},
                {"type": "image", "image": image},
                {"type": "text", "text": "프레임 1/4 (3초 전 프레임)"},
                {"type": "image", "image": image},
                {"type": "text", "text": "프레임 2/4 (2초 전 프레임)"},
                {"type": "image", "image": image},
                {"type": "text", "text": "프레임 3/4 (1초 전 프레임)"},
                {"type": "image", "image": image},
                {"type": "text", "text": "프레임 4/4 (현재 프레임)"},
                {"type": "text", "text": INSTRUCTION_QUERY},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)


print("Warming up (torch.compile takes time on first runs)...")
with torch.inference_mode():
    for w in range(3):
        t_w = time.time()
        warm_inputs = build_inputs(img, instruction)
        _ = model.generate(**warm_inputs, max_new_tokens=MAX_NEW_TOKENS)
        torch.cuda.synchronize()
        print(f"  warmup {w + 1}/3: {time.time() - t_w:.2f}s")

for i in range(N_ITERS):
    t0 = time.time()
    inputs = build_inputs(img, instruction)
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    torch.cuda.synchronize()
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    out = processor.parse_response(response)
    end = time.time()
    new_tokens = outputs.shape[-1] - input_len
    print(type(out), out.keys())
    print(out.get("content"))
    print(f"time: {end - t0:.2f}s  ({new_tokens} tok, {new_tokens / (end - t0):.1f} tok/s)")
