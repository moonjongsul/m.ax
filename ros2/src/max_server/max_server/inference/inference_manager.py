"""InferenceManager: pure-Python policy loader & predictor (no ROS deps)."""

from pathlib import Path

import numpy as np


DTYPE_MAP = {
    "float32": "float32",
    "bfloat16": "bfloat16",
    "float16": "float16",
}


LEROBOT_POLICY_MAP = {
    "pi0.5": "lerobot.policies.pi05.modeling_pi05.PI05Policy",
    "smolvla": "lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy",
}


class InferenceManager:

    def __init__(self):
        self.framework: str | None = None
        self.policy_name: str | None = None
        self.checkpoint: str | None = None
        self.device: str | None = None
        self.dtype: str | None = None

        self.policy = None
        self._preprocessor = None
        self._postprocessor = None

    # ─── Load / Unload ────────────────────────────────────────────────────────

    def load(self, framework: str, policy: str, checkpoint: str,
             device: str, dtype: str) -> tuple[bool, str]:
        try:
            if framework != "lerobot":
                return False, f"Unsupported framework: {framework}"
            if policy not in LEROBOT_POLICY_MAP:
                return False, (
                    f"Unsupported policy '{policy}'. "
                    f"Available: {list(LEROBOT_POLICY_MAP)}"
                )
            if dtype not in DTYPE_MAP:
                return False, f"Unsupported dtype '{dtype}'. Available: {list(DTYPE_MAP)}"

            ckpt_path = self._resolve_checkpoint(checkpoint)

            import importlib
            import torch

            module_path, class_name = LEROBOT_POLICY_MAP[policy].rsplit(".", 1)
            module = importlib.import_module(module_path)
            policy_cls = getattr(module, class_name)

            policy_obj = policy_cls.from_pretrained(ckpt_path)
            policy_obj.to(device)
            policy_obj.eval()

            from lerobot.policies.factory import make_pre_post_processors
            pre, post = make_pre_post_processors(
                policy_cfg=policy_obj.config,
                pretrained_path=ckpt_path,
            )

            self.framework = framework
            self.policy_name = policy
            self.checkpoint = ckpt_path
            self.device = device
            self.dtype = dtype
            self.policy = policy_obj
            self._preprocessor = pre
            self._postprocessor = post
            return True, f"Policy {policy} loaded from {ckpt_path}"

        except Exception as e:
            self.unload()
            return False, f"Policy load failed: {e}"

    def unload(self):
        self.policy = None
        self._preprocessor = None
        self._postprocessor = None
        self.framework = None
        self.policy_name = None
        self.checkpoint = None
        self.device = None
        self.dtype = None

    def is_loaded(self) -> bool:
        return self.policy is not None

    def reset(self):
        if self.policy is not None and hasattr(self.policy, "reset"):
            self.policy.reset()

    # ─── Predict ──────────────────────────────────────────────────────────────

    def predict(
        self,
        images: dict[str, np.ndarray],
        state: np.ndarray,
        task_instruction: str,
    ) -> np.ndarray:
        if self.policy is None:
            raise RuntimeError("Policy not loaded")

        import torch

        obs: dict = {
            "observation.state": torch.from_numpy(np.asarray(state)).float().unsqueeze(0),
            "task": task_instruction,
        }
        for name, img_rgb in images.items():
            t = torch.from_numpy(img_rgb).float() / 255.0
            obs[f"observation.images.{name}"] = t.permute(2, 0, 1).unsqueeze(0)

        with torch.inference_mode():
            pre = self._preprocessor(obs) if self._preprocessor else obs
            action = self.policy.select_action(pre)
            out = self._postprocessor(action) if self._postprocessor else action

        return out.detach().to("cpu").numpy().squeeze(0)

    # ─── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_checkpoint(checkpoint: str) -> str:
        ckpt = checkpoint.strip()
        if not ckpt:
            raise ValueError("checkpoint is empty")
        p = Path(ckpt)
        if ckpt.startswith(("./", "../", ".", "/")) or p.exists():
            return str(p.resolve())
        return ckpt  # treat as HuggingFace repo id
