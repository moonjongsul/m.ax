import sys
sys.path.insert(0, "/workspace/m.ax/thirdparty/lerobot/src")
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(
    repo_id="moonjongsul/flip_object",
    root="/workspace/m.ax/datasets/flip_object",
)
ds.push_to_hub(private=True, tags=["robotics", "franka_fr3"])
