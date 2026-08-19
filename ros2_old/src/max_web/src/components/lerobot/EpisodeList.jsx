import { useDispatch, useSelector } from 'react-redux'
import { selectEpisodeThunk } from '../../features/lerobot/lerobotSlice'

function fmtDuration(sec) {
  const m = Math.floor(sec / 60)
  const s = (sec - m * 60).toFixed(1)
  return `${m}:${s.padStart(4, '0')}`
}

export default function EpisodeList() {
  const dispatch = useDispatch()
  const { episodes, selectedEpisode, episodeStatus, trims } = useSelector((s) => s.lerobot)

  if (!episodes.length) {
    return (
      <div className="p-3 text-sm text-gray-500">
        Load a dataset to see episodes.
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">
      <div className="px-3 py-2 text-xs font-semibold text-gray-500 border-b
                      border-gray-200 sticky top-0 bg-white">
        EPISODES ({episodes.length})
      </div>
      <ul className="flex-1 overflow-y-auto">
        {episodes.map((ep) => {
          const active = ep.episode_index === selectedEpisode
          const loading = active && episodeStatus === 'loading'
          const taskPreview = (ep.tasks?.[0] || '').slice(0, 60)
          return (
            <li
              key={ep.episode_index}
              onClick={() => !loading && dispatch(selectEpisodeThunk(ep.episode_index))}
              className={`px-3 py-2 border-b border-gray-100 cursor-pointer text-sm
                          ${active ? 'bg-blue-50 border-l-4 border-l-blue-500'
                                   : 'hover:bg-gray-50 border-l-4 border-l-transparent'}`}
            >
              <div className="flex justify-between items-baseline">
                <span className="font-mono font-semibold flex items-center gap-1">
                  ep {String(ep.episode_index).padStart(3, '0')}
                  {trims[ep.episode_index] && (
                    <span
                      title="trim set"
                      className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-500"
                    />
                  )}
                </span>
                <span className="text-xs text-gray-500">
                  {ep.length}f · {fmtDuration(ep.duration)}
                </span>
              </div>
              {taskPreview && (
                <div className="text-xs text-gray-600 truncate mt-0.5">
                  {taskPreview}
                </div>
              )}
              {loading && (
                <div className="text-xs text-blue-600 mt-0.5">loading…</div>
              )}
            </li>
          )
        })}
      </ul>
    </div>
  )
}
