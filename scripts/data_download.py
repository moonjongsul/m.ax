from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="moonjongsul/manufacturing_kitting_dataset",
    repo_type="dataset",
    local_dir="/workspace/m.ax/datasets/manufacturing_kitting_dataset",
)