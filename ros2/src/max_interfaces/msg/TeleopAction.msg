# The command a teleop operator issued, echoed as a topic so a recorder
# can capture it alongside the observations it is already subscribed to.
#
# The arm is driven through xarm_api *services*, which a recorder cannot
# subscribe to. This message is that command republished: one frame of
# the action stream, carrying whatever the teleop node just sent to the
# robot and the gripper.
#
# Published every control cycle while teleop is active -- including the
# cycles where the operator is holding still. A learner needs the zero
# actions as much as the moving ones, so the stream is a fixed-rate
# signal, not an event log.

# Stamped when the command was sent, not when the recorder saw it. This
# is the axis every other recorded stream is aligned against.
std_msgs/Header header

# Which command space the arm was driven in for this frame.
#
# CARTESIAN_VELOCITY : the teleop jog. `cartesian_velocity` is the
#                      command; `joint_position` is empty.
# JOINT_POSITION     : a location move (set_servo_angle or gohome), which
#                      the operator triggers with a button.
#                      `joint_position` is the goal; `cartesian_velocity`
#                      is zero. One frame per move, at the moment the
#                      move is commanded -- the arm keeps moving after it
#                      under the controller's own trajectory.
# IDLE               : teleop is not driving the arm (not started,
#                      stopped, e-stopped, or a joy timeout). Both
#                      command fields are zero/empty. The frame still
#                      carries the gripper state and the flags, so the
#                      stream stays continuous across a pause.
uint8 CARTESIAN_VELOCITY = 0
uint8 JOINT_POSITION     = 1
uint8 IDLE               = 2

uint8 space

# [vx, vy, vz, wx, wy, wz] exactly as sent to vc_set_cartesian_velocity:
# linear in mm/s, angular in rad/s. Millimetres because that is the
# controller's unit -- converting here would mean the recorded action no
# longer matches the command that produced the motion.
float64[6] cartesian_velocity

# Frame the velocity is expressed in: false = robot base, true = tool.
# The operator toggles this mid-episode, so it belongs in every frame --
# the same velocity vector means two different motions depending on it.
bool is_tool_coord

# Joint goal in radians, only populated when space == JOINT_POSITION.
# Length is the arm's joint count (7 on an xArm7).
float64[] joint_position

# Last gripper width commanded, 0.0 = closed .. 1.0 = open, matching the
# Float32 the node publishes to the gripper. Held between presses rather
# than pulsed on the edge, so any single frame describes the full
# commanded state of the cell.
float32 gripper_width_percent

# Speed scale in effect: the operator's high/low toggle. Included because
# the raw velocity alone does not say whether the operator was at the top
# of the low range or partway through the high one.
bool high_speed

# Locks that zeroed part of the command before it was sent. The
# suppressed axes are already zero in `cartesian_velocity`; these say the
# operator asked for the lock rather than simply holding those axes
# still.
bool lock_rotation
bool lock_translation
