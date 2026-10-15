# What the task manager has decided, published for the policy node to act on.
#
# This is the only thing that tells the policy *what* to do. The policy owns
# the 30 Hz loop and the robot; the manager owns the plan. Latched
# (transient_local) so a policy node that starts late learns the current
# prompt immediately instead of idling.
#
# Published on every change AND periodically (~5 Hz) while a task runs. The
# periodic beat is a deadman: the policy stops driving the arm if this topic
# goes silent for longer than its `task_state_timeout_s`. Without it a
# manager crash leaves the arm executing the last prompt forever.

std_msgs/Header header

# Monotonic counter, incremented on every change to the fields below.
#
# The policy echoes the seq it used in PolicyState.task_seq. The manager
# needs that echo because `head_prompt: self` makes a head read depend on
# the current subtask -- a read computed under the previous prompt must not
# be allowed to advance the plan.
uint32 seq

# ACTIVE : a plan is running; `policy_prompt` is what the policy should run.
# IDLE   : no task. The policy should hold zero velocity.
# DONE   : the main prompt's task finished. Terminal; see the manager's
#          task_complete topic for the report.
# BLOCKED: the manager needs an operator (recognizer abstained, retry budget
#          exhausted). The policy holds zero velocity, same as IDLE, but the
#          distinction matters to a UI.
# PROBING: no plan yet. The policy holds zero velocity. If `head_prompt` is
#          set it still reads the heads with it -- that is the head-based
#          start, where the subtask head's approach_flip vs approach_pick
#          probabilities decide the initial state before any plan exists.
#          With `head_prompt` empty the manager is waiting on recognizer
#          verdicts and the policy has nothing to do.
uint8 ACTIVE  = 0
uint8 IDLE    = 1
uint8 DONE    = 2
uint8 BLOCKED = 3
uint8 PROBING = 4

uint8 state

# Goal name the planner was asked for: kit | pick | place. Empty when IDLE.
string goal

# The full prompt string to run the policy with, already assembled by the
# manager's PromptBuilder -- "Task: <neutral goal>. Subtask: <current>.
# Quality: 5. Mistake: false."
#
# Assembled manager-side, not here, because the sentence wording has to match
# what the checkpoint was trained on and that mapping lives with the planner.
string policy_prompt

# Prompt to read the MVLA heads with. Equals `policy_prompt` when the config
# says head_prompt=self, or the goal-only sentence when it says blind.
# Empty for SmolVLA, which has no heads.
string head_prompt

# Current subtask short name (approach_flip | flip | approach_pick | pick |
# move | place), and the rest of the plan including it. Informational for the
# policy; the prompt is what it acts on. A UI reads these.
string subtask
string[] remaining

# When the current subtask began, on the same clock as `header.stamp`.
#
# The policy derives `observation.elapsed_subtask` from this locally at 30 Hz.
# Sending the elapsed value instead would make it a 5 Hz round trip and the
# policy would feed a stale number into every chunk between updates.
builtin_interfaces/Time subtask_started_at

# Speed bin to put in the prompt, or -1 to omit the field (the default --
# the trained-on dropout form).
int32 speed_bin

# True once `remaining` has one entry left. The manager's completion rules
# only apply on the last subtask, and a UI shows it differently.
bool on_last_subtask
