TASK1=$(seq -s, 0 105)
TASK2=$(seq -s, 106 182)

lerobot-edit-dataset \
    --root /workspace/m.ax/datasets/manufacturing_kitting_dataset \
    --repo_id manufacturing_kitting_dataset \
    --operation.type split \
    --operation.splits "{\"flip_object\": [$TASK1], \"kit_object\": [$TASK2]}" \
    --push_to_hub=false

# split 저장 경로: ~/.cache/huggingface/lerobot/