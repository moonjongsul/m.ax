# Periodic robot/gripper telemetry broadcast on /control/robot_state_package.
# Aggregates latest observed states and last-published commands so clients
# (web UI) can subscribe to a single topic instead of many.

std_msgs/Header header

# ── Observed ───────────────────────────────────────────────────────────────
# Robot joint positions (from subscribed joint_state topic).
string[] joint_names
float64[] joint_positions

# Current end-effector pose (from subscribed current_pose topic, quat).
# position: x, y, z in meters
# orientation: qx, qy, qz, qw
bool current_pose_valid
float64 current_pose_x
float64 current_pose_y
float64 current_pose_z
float64 current_pose_qx
float64 current_pose_qy
float64 current_pose_qz
float64 current_pose_qw

# Gripper state (from subscribed gripper_state topic).
bool gripper_state_valid
float64 gripper_state

# ── Commanded (last published by server) ───────────────────────────────────
bool joint_command_valid
float64[] joint_command

bool goal_pose_valid
float64 goal_pose_x
float64 goal_pose_y
float64 goal_pose_z
float64 goal_pose_qx
float64 goal_pose_qy
float64 goal_pose_qz
float64 goal_pose_qw

bool gripper_command_valid
float64 gripper_command
