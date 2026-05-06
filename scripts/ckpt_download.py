from huggingface_hub import snapshot_download

DOWNLOAD_DIR = '/workspace/m.ax/checkpoints'
REPO_ID = 'moonjongsul/manufacturing_kitting_smolvla_rot6d_260430_v2'


def main():
    local_dir = f"{DOWNLOAD_DIR}/{REPO_ID.split('/')[-1]}"
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type='model',
        local_dir=local_dir,
    )
    print(f"Downloaded to: {path}")


if __name__ == '__main__':
    main()