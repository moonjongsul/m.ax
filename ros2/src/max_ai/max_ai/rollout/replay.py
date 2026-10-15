"""ROS 없이 rollout 파이프라인 전체를 녹화 에피소드로 돌려본다.

    python -m max_ai.rollout.replay --config config/xarm7_kitting.yaml \
        --episode 33 [--latency 0.1] [--speed-scale 1.0]

노드 네 개가 하는 일을 한 프로세스에서 가상 시계로 엮는다. 노드가 쪼개진 뒤에도
제어 로직은 전부 ROS 없이 돌기 때문에 이게 가능하고, 이 경로가 살아 있는 한
체크포인트를 바꿀 때마다 로봇 없이 먼저 확인할 수 있다.

    policy    PolicyExecutor + SimulatedWorker
    manager   EpisodeManager (SubtaskTracker + TaskManager)
    planner   LocalPlanSource (rules.py 직접 호출)
    인식기     SyncRecognizerWorker + CaptureScheduler

**한 틱 지연을 일부러 넣는다.** 실기에서는 헤드 읽기가 policy -> manager ->
TaskState -> policy로 한 바퀴 돌기 때문에, 헤드로 subtask가 전진해도 새 프롬프트는
다음 틱에 쓰인다. 노드로 쪼개기 전에는 같은 틱 안에서 됐다. replay가 그 지연을
빼먹으면 실기보다 나은 조건을 재는 셈이 된다.

**개루프다.** 로봇이 명령대로 움직이지 않으므로 누적 오차는 재지 못한다.
재는 것은 파이프라인이 맞는지다: 단위·프레임·카메라 매핑·청크 타이밍·
안전 제한이 틀리면 명령과 녹화된 액션의 차이가 눈에 띄게 커진다. MVLA면
subtask 추적이 정답 경계를 따라가는지도 본다.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from .config import load_config
from .episode import EpisodeManager, TaskStateView
from .executor import ACTIVE, HALT, IDLE, PROBING, PolicyExecutor, PromptSpec, SimulatedWorker
from .observation import RawObservation
from .policy_runner import PolicyRunner
from .recognition import CaptureScheduler
from .subtask import load_prompt_config

logger = logging.getLogger("max_ai.replay")

_MODE = {"active": ACTIVE, "probing": PROBING, "idle": IDLE, "done": HALT, "blocked": HALT}


def to_prompt(view: TaskStateView, received_at: float) -> PromptSpec:
    """TaskStateView -> PromptSpec. 실기에서 TaskState 메시지가 하는 변환과 같다."""
    return PromptSpec(
        mode=_MODE.get(view.state, IDLE),
        seq=view.seq,
        policy_prompt=view.policy_prompt,
        head_prompt=view.head_prompt or None,
        subtask_started_at=view.subtask_started_at,
        received_at=received_at,
    )


def load_episode(repo_id: str, episode: int, cameras: list[str], jpeg_quality: int = 90):
    """에피소드 프레임을 RawObservation 목록으로. 이미지는 JPEG 왕복."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, episodes=[episode], video_backend="torchcodec")
    frames, actions, subtasks, task = [], [], [], ""
    for i in range(len(ds)):
        item = ds[i]
        task = task or item["task"]
        images = {}
        for name in cameras:
            rgb = (item[f"observation.images.{name}"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            images[name] = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        # observation.state (18): [eef_position(3) rot6d(6) joint(7) gripper effort]
        # `state_vector`가 여기서 다시 같은 벡터를 만들어 내도록 되돌린다.
        # rot6d는 쿼터니언으로 되돌릴 필요가 없다 -- eef_quat에서 다시 계산되는데,
        # 데이터셋에 쿼터니언 열이 따로 있으므로 그것을 쓴다.
        state = item["observation.state"].numpy()
        quat = item["observation.state.eef_quat"].numpy()
        # eef_twist는 state에 없다. 안전 제한이 쓰지 않고, 재생은 state만
        # 재구성하면 되므로 0으로 둔다.
        frames.append(RawObservation(
            stamp=float(item["frame_index"]) / ds.fps,
            images=images,
            eef_position=state[:3].astype(np.float64),
            eef_quat=quat[3:7].astype(np.float64),
            joint_position=state[9:16].astype(np.float64),
            gripper_position=float(state[16]),
            gripper_effort_raw=float(state[17]) * 600.0,  # 녹화값은 정규화돼 있다
        ))
        actions.append(item["action"].numpy())
        subtasks.append(int(item["subtask_index"]) if "subtask_index" in item else -1)
    return ds.fps, frames, np.asarray(actions), np.asarray(subtasks), task


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None, help="config의 체크포인트 대신")
    p.add_argument("--dataset", default="moonjongsul/xarm7-kitting-260923")
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--task", default=None, help="MVLA 목표 / SmolVLA 문장. 기본: 에피소드 과제에서")
    p.add_argument("--latency", type=float, default=None, help="가상 추론 지연(초). 기본은 실제 계산 시간")
    p.add_argument("--speed-scale", type=float, default=1.0,
                   help="녹화 액션과 비교하려면 1.0 (실기 설정의 0.5를 덮어쓴다)")
    p.add_argument("--recognizer", default=None,
                   help="인식기 디렉토리 (config 대신). 'none'이면 인식기 없이")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    cfg = load_config(args.config, overrides={"safety": {"speed_scale": args.speed_scale}})
    if args.checkpoint:
        cfg.policy.checkpoint = args.checkpoint
    if args.recognizer is not None:
        cfg.recognizer.path = "" if args.recognizer == "none" else args.recognizer

    runner = PolicyRunner(cfg.policy.checkpoint, cfg.policy.device, cfg.policy.num_inference_steps)
    fps, frames, rec_actions, gt_subtask, rec_task = load_episode(args.dataset, args.episode, runner.camera_names)
    if abs(fps - cfg.control.fps) > 1e-6:
        raise ValueError(f"dataset fps {fps} != control fps {cfg.control.fps}")

    task = args.task
    if task is None:
        if runner.is_mvla:
            from lerobot.policies.mvla.data.derive import TASK_GOALS

            task = TASK_GOALS[rec_task]
        else:
            task = rec_task

    # ── 인식기 (별도 노드가 하는 일) ──────────────────────────────────
    recog = None
    scheduler = CaptureScheduler(cfg.recognizer.capture_tolerance_s)
    if runner.is_mvla and cfg.recognizer.path:
        from .recognition import Recognizer, SyncRecognizerWorker

        recog = SyncRecognizerWorker(Recognizer(cfg.recognizer.path, cfg.recognizer.device,
                                                tuple(cfg.ros.image_size), cfg.ros.image_rotate))

    cursor = {"i": 0}

    def obs_source(_with_images: bool):
        return frames[cursor["i"]]

    # ── 정책측 / 매니저측 ─────────────────────────────────────────────
    worker = SimulatedWorker(runner, cfg, latency_s=args.latency)
    policy = PolicyExecutor(cfg, runner, worker, obs_source, cfg.control.task_state_timeout_s)
    manager = EpisodeManager(
        cfg, is_mvla=runner.is_mvla,
        prompt_config=load_prompt_config(cfg.policy.checkpoint) if runner.is_mvla else None,
        recognizer_available=recog is not None,
    )
    cfg.control.max_episode_s = len(frames) / fps + 5.0

    def apply(events, now):
        """인식기 예약 이벤트를 스케줄러에 반영. 실기에서는 서비스 호출이 된다."""
        for ev in events:
            if ev.kind == "cancel_recognition":
                scheduler.cancel(ev.data["purposes"])
            elif ev.kind == "schedule_recognition":
                rid, n, _late = scheduler.schedule(
                    ev.data["purpose"], ev.data["reference"], ev.data["offsets_s"], now,
                    cancel_pending=ev.data["cancel_pending"],
                )
                if n and manager.tracker is not None:
                    manager.tracker.note_request(ev.data["purpose"], rid)

    first = frames[0]
    policy.begin(0.0, cfg.control.max_episode_s)
    apply(manager.begin(0.0, goal=task if runner.is_mvla else "",
                        task_instruction="" if runner.is_mvla else task,
                        gripper_effort=first.effort_normalized(cfg.ros.gripper_effort_max)), 0.0)
    # 한 틱 지연: 정책은 지난 틱의 TaskState로 움직인다 (실기의 왕복).
    prompt = to_prompt(manager.task_state(0.0), 0.0)

    t_wall = time.time()
    cmds, acts, events_log = [], [], []
    for i in range(len(frames)):
        cursor["i"] = i
        t = i / fps
        raw = frames[i]
        effort = raw.effort_normalized(cfg.ros.gripper_effort_max)

        tick = policy.tick(t, prompt)
        cmds.append(tick.command.twist if tick.command is not None else np.zeros(6))
        acts.append(policy.log[-1]["action"] if policy.log else None)

        events = []
        for head, t_obs, seq in tick.heads:
            events += manager.on_head(head, t_obs, seq)
        events += manager.on_tick(raw.gripper_position, t)

        # 인식기: 예약 시각이 된 캡처를 뜨고, 돌아온 판정을 매니저에 넣는다
        if recog is not None:
            for cap in scheduler.pop_due(t):
                recog.submit(cap, raw, t)
            for v in recog.poll(t):
                events += manager.on_verdict(v, effort, t)
        apply(events, t)
        for ev in events:
            if ev.kind not in ("schedule_recognition", "cancel_recognition"):
                events_log.append((round(t, 3), f"{ev.kind}: {ev.detail}"))

        prompt = to_prompt(manager.task_state(t), t)
        if not manager.active:
            logger.info(f"manager ended at frame {i}: {manager.status} ({manager.reason})")
            break
        if not policy.active:
            logger.info(f"policy ended at frame {i}: {policy.status} ({policy.reason})")
            break
    n = len(cmds)
    cmds = np.asarray(cmds)

    # 정책 액션(안전 제한 전)과 녹화 액션 비교. 첫 청크 전 구간은 뺀다.
    have = np.array([a is not None for a in acts])
    pol = np.array([a if a is not None else [np.nan] * 7 for a in acts], dtype=np.float64)
    rec = rec_actions[:n]
    names = ["vx", "vy", "vz", "wx", "wy", "wz", "gripper"]
    rmse = {k: float(np.sqrt(np.nanmean((pol[have, j] - rec[have, j]) ** 2))) for j, k in enumerate(names)}
    corr = {k: float(np.corrcoef(pol[have, j], rec[have, j])[0, 1]) for j, k in enumerate(names[:6])}
    summary = {
        "episode": args.episode, "task": task, "frames": n,
        "first_action_frame": int(np.argmax(have)) if have.any() else -1,
        "policy_status": policy.status, "policy_reason": policy.reason,
        "manager_status": manager.status, "manager_reason": manager.reason,
        "mean_latency_s": float(np.mean([r["latency_s"] for r in policy.log])),
        "rmse_policy_vs_recorded": rmse, "corr_policy_vs_recorded": corr,
        "safety_flags": {f: int(sum(f in r["flags"] for r in policy.log))
                         for f in ("clip", "accel", "workspace", "stop", "invalid_action",
                                   "stale_observation")},
        "wall_s": time.time() - t_wall,
    }
    if runner.is_mvla:
        from .subtask import SUBTASK_NAMES

        tracked = np.array([SUBTASK_NAMES.index(r["subtask"]) if r.get("subtask") else -1
                            for r in _subtask_trace(policy, manager, n)])
        lab = gt_subtask[:n] >= 0
        summary["subtask_tracked_acc"] = float((tracked[lab] == gt_subtask[:n][lab]).mean())
        summary["events"] = events_log
        summary["completion"] = manager.completion

    out = Path(args.out or f"replay_ep{args.episode}.json")
    out.write_text(json.dumps({"summary": summary, "log": policy.log,
                               "recorded_action": rec.tolist(), "recorded_subtask": gt_subtask[:n].tolist()},
                              indent=1, default=float))
    print(json.dumps(summary, indent=2, default=float))
    print(f"saved {out}")


def _subtask_trace(policy: PolicyExecutor, manager: EpisodeManager, n: int) -> list[dict]:
    """틱별 subtask 이름. 정책 로그의 프롬프트에서 역으로 읽는다.

    정책 로그에는 subtask 이름이 없다 -- 정책은 프롬프트만 받는다. 매니저가
    틱마다 기록을 남기지 않으므로, 프롬프트에 박힌 문장으로 되돌린다.
    """
    from .subtask import SUBTASK_SENTENCES

    by_sentence = {v: k for k, v in SUBTASK_SENTENCES.items()}
    trace = []
    for row in policy.log[:n]:
        name = None
        for sentence, short in by_sentence.items():
            if f"Subtask: {sentence}." in (row.get("prompt") or ""):
                name = short
                break
        trace.append({"subtask": name})
    while len(trace) < n:
        trace.append({"subtask": None})
    return trace


if __name__ == "__main__":
    main()
