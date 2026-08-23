# Recorder state, published continuously so an operator (and the teleop
# node) can tell at a glance whether data is being captured.
#
# Publish on every state change AND periodically (~1 Hz) while idle. The
# periodic beat is what lets a subscriber notice that the recorder --
# often on another machine -- has gone away: silence for more than a few
# seconds means "unknown", not "idle". Latch it (transient_local) so a
# node that starts late learns the state immediately.

std_msgs/Header header

# IDLE      : nothing buffered. DELETE here undoes the last SAVE.
# RECORDING : capturing.
# PENDING   : capture stopped, data still in memory, waiting for the
#             operator to commit with SAVE or drop it with DELETE.
# SAVING    : writing to storage. Transient; START is refused until it
#             returns to IDLE. Recorders that save fast enough to do it
#             inside the service call never report this.
uint8 IDLE      = 0
uint8 RECORDING = 1
uint8 PENDING   = 2
uint8 SAVING    = 3

uint8 state

# Episode being recorded, held in PENDING, or being written in SAVING.
# While IDLE this is the most recently saved episode -- i.e. the one a
# DELETE would undo -- or empty if there is nothing to undo.
string episode_id

# Length of the current or just-finished episode, in seconds.
float32 duration

# Frames captured for that episode. What counts as a frame is the
# recorder's business; it exists so the operator can see the count moving
# and catch a stalled subscription.
uint32 frame_count

# Optional detail for a UI: last error, storage path, free disk space.
string message