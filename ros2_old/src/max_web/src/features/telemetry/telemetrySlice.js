import { createSlice } from '@reduxjs/toolkit'

const initialState = {
  jointNames: [],
  jointPositions: [],
  currentPose: null, // { x, y, z, qx, qy, qz, qw }
  gripperState: null,
  jointCommand: [],
  goalPose: null,
  gripperCommand: null,
  lastUpdate: 0,
}

const telemetrySlice = createSlice({
  name: 'telemetry',
  initialState,
  reducers: {
    setRobotStates(state, action) {
      const s = action.payload
      state.jointNames = s.joint_names || []
      state.jointPositions = s.joint_positions || []
      state.currentPose = s.current_pose_valid
        ? {
            x: s.current_pose_x, y: s.current_pose_y, z: s.current_pose_z,
            qx: s.current_pose_qx, qy: s.current_pose_qy,
            qz: s.current_pose_qz, qw: s.current_pose_qw,
          }
        : null
      state.gripperState = s.gripper_state_valid ? s.gripper_state : null
      state.jointCommand = s.joint_command_valid ? (s.joint_command || []) : []
      state.goalPose = s.goal_pose_valid
        ? {
            x: s.goal_pose_x, y: s.goal_pose_y, z: s.goal_pose_z,
            qx: s.goal_pose_qx, qy: s.goal_pose_qy,
            qz: s.goal_pose_qz, qw: s.goal_pose_qw,
          }
        : null
      state.gripperCommand = s.gripper_command_valid ? s.gripper_command : null
      state.lastUpdate = Date.now()
    },
  },
})

export const { setRobotStates } = telemetrySlice.actions
export default telemetrySlice.reducer
