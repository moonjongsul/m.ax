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
  expressionType: '',
  step: 0,
  lastAction: [],

  // form state
  form: {
    framework: 'lerobot',
    policy: 'smolvla',
    checkpoint:
      '/workspace/m.ax/checkpoints/local_260410/smolvla_kitting_scratch_b32/checkpoints/082000/pretrained_model',
    device: 'cuda',
    dtype: 'bfloat16',
    taskInstruction: 'pick part and flip part',
    expressionType: 'rot6d',
  },
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
      state.expressionType = s.expression_type || ''
      state.step = s.step || 0
    },
    setFeedback(state, action) {
      const f = action.payload
      state.step = f.step ?? state.step
      state.lastAction = f.last_action || []
    },
    setFormField(state, action) {
      state.form[action.payload.key] = action.payload.value
    },
  },
})

export const { setStatus, setFeedback, setFormField } = inferenceSlice.actions
export default inferenceSlice.reducer
