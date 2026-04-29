import { useMemo, useRef } from 'react'
import { useSelector } from 'react-redux'
import VideoGrid from './VideoGrid'
import SeriesPlot from './SeriesPlot'

function CurrentValuesTable({ names, row, label }) {
  if (!row || !names?.length) return null
  return (
    <div className="text-xs font-mono">
      <div className="font-semibold text-gray-600 mb-1">{label} @ playhead</div>
      <div className="grid grid-cols-4 gap-x-3 gap-y-0.5">
        {names.map((n, i) => (
          <div key={n} className="flex justify-between">
            <span className="text-gray-500">{n}:</span>
            <span>{Number(row[i]).toFixed(3)}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function EpisodeViewer() {
  const primaryVideoRef = useRef(null)
  const { info, frames, playhead, episodeStatus, episodeError } = useSelector(
    (s) => s.lerobot,
  )

  const frameIdx = useMemo(() => {
    if (!frames) return 0
    const i = Math.round(playhead * frames.fps)
    return Math.max(0, Math.min(frames.length - 1, i))
  }, [playhead, frames])

  if (!frames && episodeStatus !== 'loading') {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-400">
        Select an episode from the list.
      </div>
    )
  }
  if (episodeStatus === 'loading') {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-500">
        Loading episode…
      </div>
    )
  }
  if (episodeError) {
    return (
      <div className="flex-1 flex items-center justify-center text-red-600 font-mono text-sm">
        {episodeError}
      </div>
    )
  }

  const tasks = frames.tasks || []
  const actionRow = frames.action?.[frameIdx]
  const stateRow = frames.state?.[frameIdx]

  return (
    <div className="flex-1 flex flex-col overflow-y-auto">
      {/* Task instructions */}
      <div className="px-3 py-2 bg-amber-50 border-b border-amber-200">
        <div className="text-xs font-semibold text-amber-800 mb-1">
          TASK INSTRUCTIONS ({tasks.length})
        </div>
        <div className="flex flex-wrap gap-1">
          {tasks.map((t, i) => (
            <span
              key={i}
              className="px-2 py-0.5 text-xs bg-white border border-amber-300
                         text-amber-900 rounded"
            >
              {t}
            </span>
          ))}
        </div>
      </div>

      {/* Videos */}
      <VideoGrid primaryRef={primaryVideoRef} />

      {/* Current frame values */}
      <div className="grid grid-cols-2 gap-3 px-3 py-2 bg-gray-50 border-y border-gray-200">
        <CurrentValuesTable
          names={info?.action_names}
          row={actionRow}
          label="action"
        />
        <CurrentValuesTable
          names={info?.state_names}
          row={stateRow}
          label="observation.state"
        />
      </div>

      {/* Plots */}
      <div className="p-3 space-y-3">
        <SeriesPlot
          data={frames.action}
          names={info?.action_names || []}
          fps={frames.fps}
          playhead={playhead}
          title="action"
        />
        <SeriesPlot
          data={frames.state}
          names={info?.state_names || []}
          fps={frames.fps}
          playhead={playhead}
          title="observation.state"
        />
      </div>
    </div>
  )
}
