import { createSlice, createAsyncThunk } from '@reduxjs/toolkit'
import * as api from './api'

export const loadDatasetThunk = createAsyncThunk(
  'lerobot/loadDataset',
  async (path, { dispatch }) => {
    const info = await api.loadDataset(path)
    const eps = await api.listEpisodes()
    return { info, episodes: eps.episodes, fps: eps.fps }
  },
)

export const selectEpisodeThunk = createAsyncThunk(
  'lerobot/selectEpisode',
  async (episodeIndex) => {
    const frames = await api.getEpisodeFrames(episodeIndex)
    return { episodeIndex, frames }
  },
)

export const startExportThunk = createAsyncThunk(
  'lerobot/startExport',
  async (_, { getState }) => {
    const { lerobot } = getState()
    return await api.exportDataset(lerobot.exportPath, lerobot.trims)
  },
)

export const refreshExportStatusThunk = createAsyncThunk(
  'lerobot/refreshExportStatus',
  async () => await api.exportStatus(),
)

const initialState = {
  // dataset-level
  path: '',
  info: null,            // { fps, video_keys, action_names, state_names, ... }
  episodes: [],          // [{ episode_index, length, duration, tasks }]
  loadStatus: 'idle',    // idle | loading | succeeded | failed
  loadError: null,

  // episode-level
  selectedEpisode: null, // episode_index
  episodeStatus: 'idle',
  episodeError: null,
  frames: null,          // { length, fps, action[][], state[][], timestamp[], tasks[], videos{} }

  // playback (current episode only — reset on switch)
  playhead: 0,
  playing: false,
  playbackRate: 1,

  // trim ranges per episode_index → { trimStart, trimEnd } in seconds.
  // Survives episode switches; cleared only when a new dataset is loaded.
  trims: {},
  // Mirror of trims[selectedEpisode] for convenient selectors / immer.
  trimStart: 0,
  trimEnd: 0,

  // export
  exportPath: '',
  exportStatus: null, // { running, progress, current_episode, total_episodes, message, error, out_path }
  exportStartError: null,
}

function syncTrim(state) {
  const t = state.trims[state.selectedEpisode]
  if (t) {
    state.trimStart = t.trimStart
    state.trimEnd = t.trimEnd
  } else {
    state.trimStart = 0
    state.trimEnd = state.frames ? state.frames.length / state.frames.fps : 0
  }
}

const slice = createSlice({
  name: 'lerobot',
  initialState,
  reducers: {
    setPath(state, { payload }) {
      state.path = payload
    },
    setExportPath(state, { payload }) {
      state.exportPath = payload
    },
    setPlayhead(state, { payload }) {
      state.playhead = payload
    },
    setTrimStart(state, { payload }) {
      const v = Math.min(payload, state.trimEnd)
      state.trimStart = v
      if (state.selectedEpisode != null) {
        const cur = state.trims[state.selectedEpisode] || { trimEnd: state.trimEnd }
        state.trims[state.selectedEpisode] = { trimStart: v, trimEnd: cur.trimEnd }
      }
    },
    setTrimEnd(state, { payload }) {
      const v = Math.max(payload, state.trimStart)
      state.trimEnd = v
      if (state.selectedEpisode != null) {
        const cur = state.trims[state.selectedEpisode] || { trimStart: state.trimStart }
        state.trims[state.selectedEpisode] = { trimStart: cur.trimStart, trimEnd: v }
      }
    },
    setPlaying(state, { payload }) {
      state.playing = payload
    },
    setPlaybackRate(state, { payload }) {
      state.playbackRate = payload
    },
    resetTrim(state) {
      const dur = state.frames ? state.frames.length / state.frames.fps : 0
      state.trimStart = 0
      state.trimEnd = dur
      if (state.selectedEpisode != null) {
        delete state.trims[state.selectedEpisode]
      }
    },
  },
  extraReducers: (b) => {
    b.addCase(loadDatasetThunk.pending, (s) => {
      s.loadStatus = 'loading'
      s.loadError = null
    })
    b.addCase(loadDatasetThunk.fulfilled, (s, { payload }) => {
      s.loadStatus = 'succeeded'
      s.info = payload.info
      s.episodes = payload.episodes
      s.selectedEpisode = null
      s.frames = null
      // Trims are dataset-scoped; clear on new dataset load.
      s.trims = {}
      s.trimStart = 0
      s.trimEnd = 0
      // Suggest a default export path next to the source.
      const src = (payload.info?.path || s.path || '').replace(/\/+$/, '')
      if (src && !s.exportPath) {
        s.exportPath = `${src}_trimmed`
      }
      s.exportStatus = null
      s.exportStartError = null
    })
    b.addCase(loadDatasetThunk.rejected, (s, { error }) => {
      s.loadStatus = 'failed'
      s.loadError = error.message
    })

    b.addCase(selectEpisodeThunk.pending, (s, { meta }) => {
      s.episodeStatus = 'loading'
      s.episodeError = null
      s.selectedEpisode = meta.arg
    })
    b.addCase(selectEpisodeThunk.fulfilled, (s, { payload }) => {
      s.episodeStatus = 'succeeded'
      s.frames = payload.frames
      s.playhead = 0
      s.playing = false
      syncTrim(s)
    })
    b.addCase(selectEpisodeThunk.rejected, (s, { error }) => {
      s.episodeStatus = 'failed'
      s.episodeError = error.message
    })

    b.addCase(startExportThunk.pending, (s) => {
      s.exportStartError = null
    })
    b.addCase(startExportThunk.fulfilled, (s) => {
      s.exportStatus = {
        running: true, progress: 0, message: 'starting…',
        current_episode: null, total_episodes: 0, error: null, out_path: '',
      }
    })
    b.addCase(startExportThunk.rejected, (s, { error }) => {
      s.exportStartError = error.message
    })
    b.addCase(refreshExportStatusThunk.fulfilled, (s, { payload }) => {
      s.exportStatus = payload
    })
  },
})

export const {
  setPath, setExportPath, setPlayhead, setTrimStart, setTrimEnd,
  setPlaying, setPlaybackRate, resetTrim,
} = slice.actions

export default slice.reducer
