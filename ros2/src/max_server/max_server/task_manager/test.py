from transformers import AutoProcessor, AutoModelForMultimodalLM
import cv2
import time

img = cv2.imread("./kitting_target_img.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(f"Image shape: {img.shape}")
MODEL_ID = "google/gemma-4-E4B-it"

# Load model
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID, 
    dtype="auto", 
    device_map="auto"
)

instruction = "이 이미지는 제조 도메인에서 로봇을 활용한 키팅 작업 수행을 위한 목표 이미지야. 구체적으로 설명하자면, 트레이에 부품이 담겨진 모습이고, \
트레이에는 보이는 것처럼 부품들이 담겨 있어. 이 이미지를 보고 트레이에 담긴 부품들의 종류와 위치를 설명해줘. ** 같은거 넣지 말고 응답은 아주 간략하게 해. "

for i in range(100):
    t0 = time.time()
    # Prompt - add image before text
    messages = [
        {
            "role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": instruction}
            ]
        }
    ]

    # Process input
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    # Generate output
    outputs = model.generate(**inputs, max_new_tokens=64)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

    result = processor.parse_response(response).get("content")
    # Parse output
    out = processor.parse_response(response)
    end = time.time()
    print(type(out), out.keys())
    print(out.get("content"))
    print(f'time: {end - t0:.2f}s')