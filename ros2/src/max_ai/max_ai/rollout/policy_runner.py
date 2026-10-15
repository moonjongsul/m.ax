"""체크포인트 로드와 추론. SmolVLA와 MVLA를 같은 인터페이스로.

전처리·후처리는 체크포인트에 저장된 파이프라인을 그대로 쓴다(카메라 이름
바꾸기, 토크나이즈, 정규화, 역정규화). 여기서 다시 구현하면 학습과 추론이
조용히 갈라진다.
"""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger("max_ai.policy")

ELAPSED_KEY = "observation.elapsed_subtask"
ACTION_KEY = "action"


@dataclass
class HeadRead:
    """MVLA 헤드 한 번 읽은 결과. 전부 파이썬 스칼라/배열."""

    subtask_probs: np.ndarray      # (7,) -- 6 라벨 + unlabelled
    value_subtask: float           # [-1, 0], 0 = 이 subtask 성공 직전
    value_subtask_spread: float
    value_episode: float
    status_probs: np.ndarray       # (3,) running / success / failure

    @property
    def subtask(self) -> int:
        return int(self.subtask_probs[:6].argmax())


# 플로우 매칭 스텝 수를 담은 속성 이름이 정책마다 다르다. SmolVLA/MVLA는
# `num_steps`, pi0/pi0.5는 `num_inference_steps`. 없는 쪽에 대입하면 조용히
# 새 속성이 생기고 설정한 스텝 수가 무시되므로, 있는 것에만 쓴다.
_STEP_ATTRS = ("num_steps", "num_inference_steps")


def inference_steps(cfg) -> int | None:
    for attr in _STEP_ATTRS:
        if hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    return None


def set_inference_steps(cfg, steps: int) -> str:
    for attr in _STEP_ATTRS:
        if hasattr(cfg, attr):
            setattr(cfg, attr, steps)
            return attr
    raise AttributeError(
        f"{type(cfg).__name__} has none of {_STEP_ATTRS}; cannot set inference steps"
    )


def resolve_checkpoint(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if (p / "pretrained_model").is_dir():
        p = p / "pretrained_model"
    if not (p / "config.json").is_file():
        raise FileNotFoundError(f"no config.json under {p}")
    return p


class PolicyRunner:
    def __init__(self, checkpoint: str | Path, device: str = "cuda", num_inference_steps: int | None = None):
        import lerobot.policies  # noqa: F401 - config 서브클래스(mvla 포함) 등록
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.path = resolve_checkpoint(checkpoint)
        self.device = device
        cfg = PreTrainedConfig.from_pretrained(self.path)
        if num_inference_steps is not None:
            set_inference_steps(cfg, int(num_inference_steps))
        # 가중치를 처음부터 목표 디바이스로 올린다. `from_pretrained`는
        # `cfg.device`를 safetensors 로더에 넘기므로, 이것을 설정하지 않으면
        # 전부 CPU로 한 번 읽고 나서 `.to(device)`로 또 한 벌 만든다.
        #
        # (이것만으로 pi0.5의 OOM이 해결되지는 않는다 -- 그쪽 피크는 로더가
        # 아니라 `to_bfloat16_for_selected_params`가 VLM을 fp32로 만든 뒤
        # 캐스팅하는 데서 온다. 실측으로 확인했다.)
        cfg.device = device
        cls = get_policy_class(cfg.type)
        t0 = time.time()
        self.policy = cls.from_pretrained(self.path, config=cfg)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=str(self.path),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.config = self.policy.config
        self.is_mvla = hasattr(self.policy, "head_outputs")
        self.camera_names = self._dataset_camera_names()
        self._check_state_dim()
        logger.info(
            f"loaded {cfg.type} from {self.path} in {time.time() - t0:.1f}s "
            f"(chunk {self.config.chunk_size}, steps {inference_steps(self.config)}, "
            f"cameras {self.camera_names})"
        )

    def _state_dim_from_normalizer(self) -> int | None:
        """정규화 통계가 기대하는 state 폭. 없으면 None.

        **`config.json`의 `input_features`를 믿지 않는다.** 학습 스크립트가
        데이터셋에서 feature를 추론해 덮어쓰기 전의 값이 그대로 저장되는
        경우가 있어서(실제로 18채널로 학습된 체크포인트에 `[6]`이 박혀
        있었다), 거기에 맞춰 거절하면 정상 체크포인트를 막는다.

        반면 정규화 통계는 학습에 쓴 데이터셋에서 계산되어 저장되므로,
        모델이 실제로 본 채널 수와 어긋날 수 없다. `state_proj`는
        `max_state_dim`(32)로 패딩된 뒤라 폭을 말해주지 않는다.
        """
        for step in getattr(self.pre, "steps", []):
            stats = getattr(step, "stats", None)
            if not isinstance(stats, dict):
                continue
            entry = stats.get("observation.state")
            if not isinstance(entry, dict):
                continue
            for key in ("mean", "std", "min", "max"):
                value = entry.get(key)
                shape = getattr(value, "shape", None)
                if shape:
                    return int(shape[-1])
        return None

    def _check_state_dim(self) -> None:
        """정규화 통계의 state 폭이 `state_vector`가 만드는 것과 같은지.

        틀렸을 때의 증상이 조용하다 -- `prepare_state`는 입력을
        `max_state_dim`(32)까지 패딩만 하고 `input_features`와 비교하지
        않으므로, 채널이 어긋나도 에러 없이 엉뚱한 값을 읽고 팔이 그럴듯하게
        틀린 방향으로 움직인다. 그래서 로드 시점에 한 번 확인하고 거절한다.

        state 레이아웃을 바꾼 뒤 구 체크포인트를 돌릴 때 실제로 걸린다.
        """
        from .observation import STATE_DIM

        width = self._state_dim_from_normalizer()
        if width is None:
            logger.warning(
                "could not read observation.state width from the checkpoint's "
                "normalizer stats; skipping the dim check"
            )
            return
        if width != STATE_DIM:
            raise ValueError(
                f"{self.path}: 체크포인트의 정규화 통계는 observation.state를 "
                f"{width}채널로 보는데 이 코드가 만드는 것은 {STATE_DIM}채널이다 "
                "([x y z] + rot6d + j1..j7 + gripper + gripper_effort). "
                "구 레이아웃으로 학습된 체크포인트이면 새 데이터셋으로 재학습할 것 "
                "-- 채널이 어긋난 채로 돌면 에러 없이 틀린 방향으로 움직인다."
            )

    def _dataset_camera_names(self) -> list[str]:
        """정책이 기대하는 카메라를 데이터셋 이름으로.

        정책 feature는 camera1/2/3이고, 저장된 rename 단계가 데이터셋
        이름 -> cameraN으로 바꾼다. 그 표를 거꾸로 읽는다.
        """
        rename = {}
        for step in getattr(self.pre, "steps", []):
            m = getattr(step, "rename_map", None)
            if m:
                rename.update(m)
        inverse = {v: k for k, v in rename.items()}
        names = []
        for key in self.config.image_features:
            src = inverse.get(key, key)
            names.append(src.removeprefix("observation.images."))
        return names

    def _autocast(self):
        """정책이 `dtype`을 선언하면 그것을 따른다.

        pi0/pi0.5는 config에 `dtype`("float32" | "bfloat16")이 있고, float32로
        학습·공개된 체크포인트를 bfloat16 autocast로 돌리면 플로우 매칭 적분이
        10스텝 내내 저정밀로 누적된다. SmolVLA/MVLA에는 이 필드가 없고 bf16이
        기본이므로 동작이 바뀌지 않는다.
        """
        if not self.device.startswith("cuda"):
            return nullcontext()
        declared = getattr(self.config, "dtype", None)
        if declared == "float32":
            return nullcontext()
        return torch.autocast("cuda", dtype=torch.bfloat16)

    def queue_is_empty(self) -> bool:
        """다음 `select_action`이 실제로 추론을 돌릴지.

        액션 큐를 어디에 두는지가 정책마다 다르다. SmolVLA/MVLA는
        `_queues["action"]`에 두고 `_check_get_actions_condition()`을
        제공하지만 pi0.5는 `_action_queue` 하나뿐이고 그 메서드가 없다 --
        그걸 호출하면 AttributeError로 죽는다. 둘 다 "비었으면 추론"이므로
        길이만 본다.

        어느 쪽도 못 찾으면 True를 돌려준다. 추론 지연 측정이 틀리는 것은
        로그가 부정확해지는 것뿐이고, 제어에는 영향이 없다.
        """
        policy = self.policy
        queue = getattr(policy, "_action_queue", None)
        if queue is None:
            queues = getattr(policy, "_queues", None)
            if isinstance(queues, dict):
                queue = queues.get("action") or queues.get(ACTION_KEY)
        return True if queue is None else len(queue) == 0

    def _batch(self, obs: dict[str, torch.Tensor], task: str, elapsed: float | None) -> dict:
        frame = dict(obs)
        frame["task"] = task
        if self.is_mvla and self.config.use_elapsed_subtask:
            frame[ELAPSED_KEY] = torch.tensor([float(elapsed or 0.0) / self.config.elapsed_scale])
        return self.pre(frame)

    @torch.no_grad()
    def infer_chunk(self, obs: dict[str, torch.Tensor], task: str, elapsed: float | None = None,
                    seed: int | None = None) -> np.ndarray:
        """(chunk_size, 7) 역정규화된 액션. m/s, rad/s, gripper 0..1."""
        batch = self._batch(obs, task, elapsed)
        noise = None
        if seed is not None:
            g = torch.Generator(device=self.device).manual_seed(seed)
            noise = torch.randn((1, self.config.chunk_size, self.config.max_action_dim),
                                generator=g, device=self.device)
        self.policy.reset()  # 큐 비우기 -- 청크 관리는 호출자(ChunkExecutor)가 한다
        with self._autocast():
            chunk = self.policy.predict_action_chunk(batch, noise=noise)
        chunk = self.post(chunk.float())
        return chunk.squeeze(0).cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def read_heads(self, obs: dict[str, torch.Tensor], task: str, elapsed: float | None) -> HeadRead | None:
        if not self.is_mvla:
            return None
        batch = self._batch(obs, task, elapsed)
        with self._autocast():
            out = self.policy.head_outputs(batch)
        heads = self.policy.heads
        return HeadRead(
            subtask_probs=out["subtask_logits"].float().softmax(-1)[0].cpu().numpy(),
            value_subtask=float(heads.value_subtask.expected_value(out["value_subtask_logits"])[0]),
            value_subtask_spread=float(heads.value_subtask.spread(out["value_subtask_logits"])[0]),
            value_episode=float(heads.value_episode.expected_value(out["value_episode_logits"])[0]),
            status_probs=out["status_logits"].float().softmax(-1)[0].cpu().numpy(),
        )
