from pathlib import Path
from omegaconf import OmegaConf, DictConfig


CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


def load_config(fname: Path, project: str) -> DictConfig:
    server_cfg = OmegaConf.load(fname)

    if project not in server_cfg.project:
        available = list(server_cfg.project.keys())
        raise ValueError(f"Unknown project '{project}'. Available: {available}")

    proj_config_file = CONFIG_DIR / server_cfg.project[project].config
    proj_cfg = OmegaConf.load(proj_config_file)

    cfg = OmegaConf.merge(server_cfg, proj_cfg)
    return cfg