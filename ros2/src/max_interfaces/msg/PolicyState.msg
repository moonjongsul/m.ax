# Everything the policy node knows, published for the manager to drive on.
#
# The manager has no sensors and no model. It reads this topic and decides
# whether the plan advances, whether a verification is due, and whether the
# task is finished. So this message carries both the head read (the manager's
# 5 Hz clock) and the fast signals it needs at tick rate (gripper position
# for release detection).
#
# Published every control tick (30 Hz) while running, and ~1 Hz while idle.
# The head fields are only fresh on the ticks a head read came back; use
# `head_seq` to tell a new read from a repeat of the last one.

std_msgs/Header header

# IDLE    : no goal accepted; not touching the arm.
# STARTING: goal accepted, waiting for the manager's first plan (or for the
#           one head read that decides the start state).
# RUNNING : executing chunks.
# STOPPED : stopped by the manager or an operator.
# ABORTED : stopped by the policy itself -- episode timeout, lost
#           observations past the grace period, or the task-state deadman.
uint8 IDLE     = 0
uint8 STARTING = 1
uint8 RUNNING  = 2
uint8 STOPPED  = 3
uint8 ABORTED  = 4

uint8 state

# Why it left RUNNING. Empty while running.
string reason

# Which TaskState.seq the latest chunk and head read were computed under.
#
# The manager drops head reads whose task_seq is not current: with
# head_prompt=self a read taken under the old subtask's prompt says nothing
# about whether the new one has advanced.
uint32 task_seq

# Loaded policy, so a UI and the manager can tell MVLA from SmolVLA without
# asking. `is_mvla` false means the head fields below are never populated and
# the manager runs open-loop on the plan (no verification, no completion).
string policy_type
string checkpoint
bool is_mvla

# ─────────────────────────────────────────────────────── chunk execution
# Index into the current chunk being executed, or -1 when none is loaded.
# Climbing past `execute_steps` means new chunks are not arriving in time.
int32 chunk_index

# Smoothed observation-to-result latency, seconds. The chunk scheduler uses
# it to request the next chunk this far early, so a UI watching it watches
# the thing that actually governs timing.
float32 latency_s

# Age of the observation this tick acted on, seconds. Past
# `stale_timeout_s` the policy holds zero velocity and says so in `flags`.
float32 observation_age_s

# ──────────────────────────────────────────────────────────── MVLA heads
# Incremented once per head read that came back. Unchanged between reads, so
# the manager can tell "a new read" from "the same read republished".
uint32 head_seq

# Softmax over the 7 subtask classes (6 labels + unlabelled), in dataset
# order: approach_flip, flip, approach_pick, pick, move, place, unlabelled.
float32[] subtask_probs

# Expected value in [-1, 0]; 0 means this subtask is about to succeed.
# `spread` is the head's own uncertainty about that.
float32 value_subtask
float32 value_subtask_spread
float32 value_episode

# Softmax over running / success / failure.
float32[] status_probs

# ───────────────────────────────────────────────── fast signals, 30 Hz
# Gripper width from the observation, 0 closed .. 1 open, and normalised
# effort 0..1.
#
# Here rather than left to the manager's own subscription because the
# manager's release detector has to see the same value the policy fed the
# model on that tick -- two independent subscriptions to the same topic can
# land on different frames, and the release instant sets the recognizer's
# capture times.
float32 gripper_position
float32 gripper_effort

# The command actually sent this tick: [vx vy vz] m/s, [wx wy wz] rad/s, in
# the robot base frame, after every safety limit.
float64[6] command_velocity

# Gripper target being held, 0 closed .. 1 open.
float32 gripper_target

# Which limits acted this tick: clip | accel | workspace | stop |
# invalid_action | stale_observation. A run that is constantly flagged is a
# run whose config does not match the cell.
string[] flags
