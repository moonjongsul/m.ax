# One state-recognizer judgement, published for the manager to act on.
#
# The recognizer is the slow, certain clock: the policy's value head says
# "this is going badly" continuously, but only this says "the part is sitting
# on the rib, not in the pocket". Retries and task completion are gated on
# these verdicts, so each one carries enough for the manager to apply its own
# hysteresis and thresholds rather than trusting a label.

std_msgs/Header header

# Why this capture was taken, echoed from the request:
#   start : deciding the initial scene state, before any plan exists.
#   flip  : checking a flip that the policy has already moved on from.
#   place : checking a placement, timed from the gripper release.
uint8 PURPOSE_START = 0
uint8 PURPOSE_FLIP  = 1
uint8 PURPOSE_PLACE = 2

uint8 purpose

# Probe that ran: mat (object pose on the mat) or tray (placement quality).
string probe

# Request id from ScheduleRecognition, so the manager can match a verdict to
# the batch it asked for and ignore ones left over from a replan.
uint32 request_id

# Which capture within that batch this is, and how many were scheduled.
# The manager aggregates once `index + 1 == total`.
uint8 index
uint8 total

# Time of the frames this verdict was computed from -- not when the compute
# finished. The capture instant is the whole point: the recognizer was
# trained on frames at fixed offsets from the boundary, and a verdict from a
# different instant is a judgement of a scene it never saw.
builtin_interfaces/Time captured_at

# Winning label. For mat: target | flipped | empty | abstain. For tray:
# ok | bad | abstain.
#
# `abstain` means the object silhouette was not found, so the crop fell back
# to the whole frame -- an input the probe was not trained on. The manager
# treats a majority of abstentions as needing an operator, not as a retry.
string label

# Full distribution, as parallel arrays. The manager thresholds on
# probs["bad"] for tray rather than on `label`, because the operating point
# is a tuning decision that belongs with the task, not with the model.
string[] prob_names
float32[] prob_values

# Out-of-distribution score from the probe. High values mean the scene does
# not look like training data at all.
float32 ood

# False when the probe found no object silhouette (see `abstain` above).
bool detected
