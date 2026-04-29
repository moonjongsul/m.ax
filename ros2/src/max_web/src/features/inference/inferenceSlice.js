import { createSlice } from '@reduxjs/toolkit'

const initialState = {
  // server state
  serverState: 'idle',      // idle | loading | ready | running | error
  detail: '',

  // currently loaded policy (mirrored from /inference/status)
  policy: {
    framework: '',
    name: '',
    checkpoint: '',
    device: '',
    dtype: '',
  },

  // currently running task
  taskInstruction: '',
  representationType: '',
  step: 0,
  lastAction: [],

  // form state — populated from /inference/status defaults on first receipt.
  // User edits afterwards are preserved; defaults are not re-applied.
  form: {
    framework: '',
    policy: '',
    checkpoint: '',
    device: '',
    dtype: '',
    taskInstruction: '',
    representationType: '',
  },
  formInitialized: false,

  // Preset poses from YAML (names only; values stay on the server).
  robotPoseNames: [],
  gripperPoseNames: [],
}

const inferenceSlice = createSlice({
  name: 'inference',
  initialState,
  reducers: {
    setStatus(state, action) {
      const s = action.payload
      state.serverState = s.server_state || 'idle'
      state.detail = s.detail || ''
      state.policy = {
        framework: s.policy_framework || '',
        name: s.policy_name || '',
        checkpoint: s.checkpoint || '',
        device: s.device || '',
        dtype: s.dtype || '',
      }
      state.taskInstruction = s.task_instruction || ''
      state.representationType = s.representation_type || ''
      state.step = s.step || 0

      state.robotPoseNames = s.robot_pose_names || []
      state.gripperPoseNames = s.gripper_pose_names || []

      // One-time form init from server-provided defaults.
      if (!state.formInitialized) {
        const hasAnyDefault =
          s.default_framework || s.default_policy || s.default_checkpoint ||
          s.default_device || s.default_dtype ||
          s.default_task_instruction || s.default_representation_type
        if (hasAnyDefault) {
          state.form = {
            framework: s.default_framework || '',
            policy: s.default_policy || '',
            checkpoint: s.default_checkpoint || '',
            device: s.default_device || '',
            dtype: s.default_dtype || '',
            taskInstruction: s.default_task_instruction || '',
            representationType: s.default_representation_type || '',
          }
          state.formInitialized = true
        }
      }
    },
    setFeedback(state, action) {
      const f = action.payload
      state.step = f.step ?? state.step
      state.lastAction = f.last_action || []
    },
    setFormField(state, action) {
      state.form[action.payload.key] = action.payload.value
      // Any user edit means the form is considered initialized; defaults
      // won't overwrite it later.
      state.formInitialized = true
    },
    resetFormToDefaults(state) {
      // Force next status receipt to re-populate form from defaults.
      state.formInitialized = false
    },
  },
})

export const { setStatus, setFeedback, setFormField, resetFormToDefaults } = inferenceSlice.actions
export default inferenceSlice.reducer
