import { createSlice } from '@reduxjs/toolkit'

const initialHost =
  typeof window !== 'undefined' ? window.location.hostname || 'localhost' : 'localhost'

const initialState = {
  host: initialHost,
  rosbridgePort: 9090,
  videoPort: 8080,
  connected: false,
  connecting: false,
  error: '',
}

const rosSlice = createSlice({
  name: 'ros',
  initialState,
  reducers: {
    setHost(state, action) { state.host = action.payload },
    setConnecting(state, action) { state.connecting = action.payload },
    setConnected(state, action) {
      state.connected = action.payload
      if (action.payload) { state.error = '' }
    },
    setError(state, action) { state.error = action.payload; state.connected = false },
  },
})

export const { setHost, setConnecting, setConnected, setError } = rosSlice.actions

export const selectRosbridgeUrl = (state) =>
  `ws://${state.ros.host}:${state.ros.rosbridgePort}`
export const selectVideoBaseUrl = (state) =>
  `http://${state.ros.host}:${state.ros.videoPort}`

export default rosSlice.reducer
