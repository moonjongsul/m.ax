TASK1=$(seq -s, 0 105)
TASK2=$(seq -s, 106 182)

DB_NAME=manufacturing_kitting_dataset

lerobot-edit-dataset \
    --root /workspace/m.ax/datasets/${DB_NAME} \
    --repo_id ${DB_NAME} \
    --operation.type split \
    --operation.splits "{\"${DB_NAME}_flip_object\": [$TASK1], \"${DB_NAME}_kit_object\": [$TASK2]}" \
    --new_root /workspace/m.ax/datasets \
    --push_to_hub=false

# split 저장 경로:
#   /workspace/m.ax/datasets/${DB_NAME}_flip_object/
#   /workspace/m.ax/datasets/${DB_NAME}_kit_object/