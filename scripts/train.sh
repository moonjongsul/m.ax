PYTORCH_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=0 \
  lerobot-train \
  --policy.type=smolvla \
  --policy.repo_id=${HF_USER}/manufacturing-kitting-smolvla \
  --policy.load_vlm_weights=true \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --steps=200000 \
  --save_freq=2000 \
  --wandb.enable=true \
  --dataset.image_transforms.enable=true \
  --dataset.repo_id=moonjongsul/manufacturing_kitting_dataset \
  --batch_size=24 \
  --output_dir=./outputs/smolvla_kitting_rot6d_b24 \
  --job_name=smolvla_kitting_rot6d_b24