import { useEffect, useState } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import {
  loadDatasetThunk, setPath, setExportPath,
  startExportThunk, refreshExportStatusThunk,
} from '../../features/lerobot/lerobotSlice'

const DEFAULT_PATH = '/workspace/m.ax/datasets/manufacturing_kitting_dataset'

function ExportRow() {
  const dispatch = useDispatch()
  const { exportPath, exportStatus, exportStartError, trims } = useSelector(
    (s) => s.lerobot,
  )

  const running = !!exportStatus?.running
  const trimmedCount = Object.keys(trims).length

  // Poll status while a job is running.
  useEffect(() => {
    if (!running) return
    const id = setInterval(() => dispatch(refreshExportStatusThunk()), 1000)
    return () => clearInterval(id)
  }, [running, dispatch])

  // Also refresh once when component mounts (in case a job is already alive).
  useEffect(() => {
    dispatch(refreshExportStatusThunk())
  }, [dispatch])

  const onExport = () => {
    if (!exportPath || running) return
    dispatch(startExportThunk())
    // Immediately fetch status so the UI flips to running.
    setTimeout(() => dispatch(refreshExportStatusThunk()), 250)
  }

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center gap-2">
        <label className="text-sm font-medium text-gray-700 whitespace-nowrap">
          Export path:
        </label>
        <input
          type="text"
          value={exportPath}
          onChange={(e) => dispatch(setExportPath(e.target.value))}
          className="flex-1 px-2 py-1 text-sm border border-gray-300 rounded
                     focus:outline-none focus:ring-2 focus:ring-emerald-400 font-mono"
          placeholder="/path/to/output_dataset"
          disabled={running}
        />
        <button
          onClick={onExport}
          disabled={running || !exportPath}
          className="px-3 py-1 text-sm bg-emerald-600 text-white rounded
                     hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed"
          title={trimmedCount === 0
            ? 'No trims set — full episodes will be exported'
            : `${trimmedCount} episode(s) have a custom trim range`}
        >
          {running ? 'Exporting…' : 'Export'}
        </button>
      </div>
      {exportStartError && (
        <div className="text-xs text-red-600 font-mono">
          {exportStartError}
        </div>
      )}
      <ExportStatusLine />
    </div>
  )
}

function ExportStatusLine() {
  const exportStatus = useSelector((s) => s.lerobot.exportStatus)
  if (!exportStatus) return null

  const running = !!exportStatus.running
  const failed = !!exportStatus.error
  const done = !running && !failed && exportStatus.progress >= 1

  if (!running && !failed && !done) return null

  let label, color, dot
  if (running) {
    const pct = (exportStatus.progress * 100).toFixed(1)
    const epPart = exportStatus.current_episode != null
      ? ` (ep ${exportStatus.current_episode}` +
        (exportStatus.total_episodes ? ` / ${exportStatus.total_episodes - 1})` : ')')
      : ''
    label = `Exporting… ${pct}%${epPart}`
    color = 'text-blue-700'
    dot = 'bg-blue-500 animate-pulse'
  } else if (failed) {
    label = `Failed: ${exportStatus.error}`
    color = 'text-red-700'
    dot = 'bg-red-500'
  } else {
    label = `Done → ${exportStatus.out_path}`
    color = 'text-emerald-700'
    dot = 'bg-emerald-500'
  }

  return (
    <div className={`flex items-center gap-2 text-sm font-medium ${color}`}>
      <span className={`inline-block w-2 h-2 rounded-full ${dot}`} />
      <span className="break-all">{label}</span>
    </div>
  )
}

export default function DatasetPathInput() {
  const dispatch = useDispatch()
  const { path, loadStatus, loadError, info } = useSelector((s) => s.lerobot)
  const [draft, setDraft] = useState(path || DEFAULT_PATH)

  const onLoad = () => {
    dispatch(setPath(draft))
    dispatch(loadDatasetThunk(draft))
  }

  const loading = loadStatus === 'loading'
  const loaded = loadStatus === 'succeeded' && info

  return (
    <div className="flex flex-col gap-2 p-3 bg-gray-50 border-b border-gray-200">
      <div className="flex items-center gap-2">
        <label className="text-sm font-medium text-gray-700 whitespace-nowrap">
          Dataset path:
        </label>
        <input
          type="text"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && !loading && onLoad()}
          className="flex-1 px-2 py-1 text-sm border border-gray-300 rounded
                     focus:outline-none focus:ring-2 focus:ring-blue-400 font-mono"
          placeholder="/path/to/lerobot_dataset"
          disabled={loading}
        />
        <button
          onClick={onLoad}
          disabled={loading || !draft}
          className="px-3 py-1 text-sm bg-blue-600 text-white rounded
                     hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? 'Loading…' : 'Load'}
        </button>
      </div>

      {loaded && <ExportRow />}

      {loadError && (
        <div className="text-xs text-red-600 font-mono">{loadError}</div>
      )}
      {loaded && (
        <div className="text-xs text-gray-600 flex flex-wrap gap-x-4">
          <span>v{info.codebase_version}</span>
          <span>robot: {info.robot_type}</span>
          <span>fps: {info.fps}</span>
          <span>episodes: {info.total_episodes}</span>
          <span>frames: {info.total_frames}</span>
          <span>videos: {info.video_keys?.length}</span>
        </div>
      )}
    </div>
  )
}
