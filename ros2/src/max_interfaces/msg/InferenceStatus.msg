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
