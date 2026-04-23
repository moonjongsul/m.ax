# Overall server state
string STATE_IDLE=idle
string STATE_LOADING=loading
string STATE_READY=ready
string STATE_RUNNING=running
string STATE_ERROR=error

string server_state
string detail

# Currently loaded policy (populated when state is ready/running)
string policy_framework
string policy_name
string checkpoint
string device
string dtype

# Current running task (populated when state is running)
string task_instruction
string expression_type
uint32 step

# Default values sourced from the server's YAML config.
# The web UI uses these to initialize form fields.
string default_framework
string default_policy
string default_checkpoint
string default_device
string default_dtype
string default_task_instruction
string default_expression_type

# Preset pose names defined in the YAML config (order preserved).
# The web UI renders one button per name.
string[] robot_pose_names
string[] gripper_pose_names
