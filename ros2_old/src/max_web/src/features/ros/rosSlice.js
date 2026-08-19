import { createSlice } from '@reduxjs/toolkit'

const initialHost =
  typeof window !== 'undefined' ? window.location.hostname || 'localhost' : 'localhost'

// Each cell is an independent ROS endpoint (its own rosbridge + web_video_server).
// 'kitting' is the primary cell (max_server); 'picking' shares domain 0.
const makeCell = (host, { rosbridgePort = 9090, videoPort = 8080 } = {}) => ({
  host,
  rosbridgePort,
  videoPort,
  connected: false,
  connecting: false,
  error: '',
})

const initialState = {
  cells: {
    kitting: makeCell(initialHost),
    // Picking defaults to the same host (max_server PC sees picking via DDS on
    // domain 0). Its vision result is consumed over rosbridge (CompressedImage
    // subscription), not web_video_server, so videoPort is unused for picking.
    // Override the host from the PickingPage input to point at the picking PC.
    picking: makeCell(initialHost),
  },
}

const rosSlice = createSlice({
  name: 'ros',
  initialState,
  reducers: {
    setHost(state, action) {
      const { cell = 'kitting', host } = action.payload
      if (state.cells[cell]) { state.cells[cell].host = host }
    },
    setConnecting(state, action) {
      const { cell = 'kitting', value } = action.payload
      if (state.cells[cell]) { state.cells[cell].connecting = value }
    },
    setConnected(state, action) {
      const { cell = 'kitting', value } = action.payload
      const c = state.cells[cell]
      if (!c) { return }
      c.connected = value
      if (value) { c.error = '' }
    },
    setError(state, action) {
      const { cell = 'kitting', error } = action.payload
      const c = state.cells[cell]
      if (!c) { return }
      c.error = error
      c.connected = false
    },
  },
})

export const { setHost, setConnecting, setConnected, setError } = rosSlice.actions

// ── Selectors (cell-aware; default to kitting for backward compatibility) ──
export const selectCell = (cell = 'kitting') => (state) =>
  state.ros.cells[cell] || state.ros.cells.kitting

export const selectRosbridgeUrl = (cell = 'kitting') => (state) => {
  const c = state.ros.cells[cell] || state.ros.cells.kitting
  return `ws://${c.host}:${c.rosbridgePort}`
}

export const selectVideoBaseUrl = (cell = 'kitting') => (state) => {
  const c = state.ros.cells[cell] || state.ros.cells.kitting
  return `http://${c.host}:${c.videoPort}`
}

export default rosSlice.reducer
