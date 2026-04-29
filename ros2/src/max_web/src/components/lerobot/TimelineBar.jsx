import { useCallback, useEffect, useRef } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import {
  setPlayhead, setPlaying, setPlaybackRate, setTrimStart, setTrimEnd, resetTrim,
} from '../../features/lerobot/lerobotSlice'

const PLAYBACK_RATES = [0.5, 1, 1.5, 2, 3]

function fmt(t) {
  if (!isFinite(t)) return '0.00'
  return t.toFixed(2)
}

export default function TimelineBar() {
  const dispatch = useDispatch()
  const { frames, playhead, trimStart, trimEnd, playing, playbackRate } = useSelector(
    (s) => s.lerobot,
  )
  const trackRef = useRef(null)
  // Drag is tracked via window-level listeners so the cursor can leave the
  // handle without losing capture. dragKindRef records what we're dragging.
  const dragKindRef = useRef(null) // 'playhead' | 'start' | 'end'
  // Latest store values, accessed inside window-level listeners (which close
  // over their first values otherwise).
  const latest = useRef({ trimStart, trimEnd, duration: 0 })

  const duration = frames ? frames.length / frames.fps : 0
  latest.current = { trimStart, trimEnd, duration }

  const xToTime = useCallback((clientX) => {
    const r = trackRef.current?.getBoundingClientRect()
    if (!r || !latest.current.duration) return 0
    const frac = Math.max(0, Math.min(1, (clientX - r.left) / r.width))
    return frac * latest.current.duration
  }, [])

  const applyAt = useCallback((clientX) => {
    const kind = dragKindRef.current
    if (!kind) return
    const t = xToTime(clientX)
    if (kind === 'playhead') {
      dispatch(setPlayhead(t))
    } else if (kind === 'start') {
      dispatch(setTrimStart(Math.min(t, latest.current.trimEnd)))
    } else if (kind === 'end') {
      dispatch(setTrimEnd(Math.max(t, latest.current.trimStart)))
    }
  }, [dispatch, xToTime])

  // Window-level listeners installed only while a drag is in progress.
  useEffect(() => {
    const onMove = (e) => {
      if (!dragKindRef.current) return
      e.preventDefault()
      applyAt(e.clientX)
    }
    const onUp = () => {
      dragKindRef.current = null
      document.body.style.userSelect = ''
      document.body.style.cursor = ''
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
    window.addEventListener('pointercancel', onUp)
    return () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
      window.removeEventListener('pointercancel', onUp)
    }
  }, [applyAt])

  const startDrag = (kind, cursor) => (e) => {
    if (!frames) return
    e.preventDefault()
    e.stopPropagation()
    dragKindRef.current = kind
    document.body.style.userSelect = 'none'
    if (cursor) document.body.style.cursor = cursor
    applyAt(e.clientX)
  }

  // Click on empty track area → move playhead (but only if we're not in the
  // middle of a different drag, and only on direct clicks of the track div).
  const onTrackPointerDown = (e) => {
    if (e.target !== trackRef.current) return // ignore clicks on handles/regions
    startDrag('playhead', 'grabbing')(e)
  }

  if (!frames) {
    return (
      <div className="px-3 py-4 text-sm text-gray-400 border-t border-gray-200">
        Timeline (no episode)
      </div>
    )
  }

  const pct = (t) => `${(t / duration) * 100}%`

  return (
    <div className="border-t border-gray-200 bg-white p-3 select-none">
      {/* Controls row */}
      <div className="flex items-center gap-3 mb-3 text-sm">
        <div className="inline-flex border border-gray-300 rounded overflow-hidden">
          {PLAYBACK_RATES.map((r) => {
            const active = r === playbackRate
            return (
              <button
                key={r}
                onClick={() => dispatch(setPlaybackRate(r))}
                className={`px-2 py-1 text-xs font-mono border-l border-gray-300 first:border-l-0
                            ${active ? 'bg-blue-600 text-white'
                                     : 'bg-white text-gray-700 hover:bg-gray-50'}`}
              >
                ×{r}
              </button>
            )
          })}
        </div>
        <button
          onClick={() => dispatch(setPlaying(!playing))}
          className="px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          {playing ? '⏸ Pause' : '▶ Play'}
        </button>
        <span className="font-mono text-gray-700">
          t = {fmt(playhead)}s / {fmt(duration)}s
        </span>
        <span className="font-mono text-gray-500">
          frame {Math.round(playhead * frames.fps)} / {frames.length}
        </span>
        <div className="flex-1" />
        <span className="font-mono text-xs text-gray-600">
          trim: [{fmt(trimStart)}, {fmt(trimEnd)}] s
          ({Math.round(trimStart * frames.fps)} – {Math.round(trimEnd * frames.fps)} f,
          {' '}len {Math.round((trimEnd - trimStart) * frames.fps)})
        </span>
        <button
          onClick={() => dispatch(resetTrim())}
          className="px-2 py-1 text-xs border border-gray-300 rounded hover:bg-gray-50"
        >
          Reset trim
        </button>
      </div>

      {/* Track */}
      <div
        ref={trackRef}
        onPointerDown={onTrackPointerDown}
        className="relative h-10 bg-gray-100 rounded cursor-pointer touch-none"
      >
        {/* keep region (between trim handles) */}
        <div
          className="absolute top-0 bottom-0 bg-blue-200/60 pointer-events-none"
          style={{ left: pct(trimStart), width: pct(trimEnd - trimStart) }}
        />
        {/* discarded regions (outside trim) */}
        <div
          className="absolute top-0 bottom-0 bg-gray-300/60 pointer-events-none"
          style={{ left: 0, width: pct(trimStart) }}
        />
        <div
          className="absolute top-0 bottom-0 bg-gray-300/60 pointer-events-none"
          style={{ left: pct(trimEnd), right: 0 }}
        />

        {/* trim start handle: thin vertical bar at boundary, ">" pointing into kept region */}
        <div
          onPointerDown={startDrag('start', 'ew-resize')}
          className="absolute top-0 bottom-0 w-4 -ml-2 z-10 cursor-ew-resize
                     touch-none flex items-center group"
          style={{ left: pct(trimStart) }}
          title={`trim start ${fmt(trimStart)}s`}
        >
          {/* exact boundary line */}
          <div className="absolute top-0 bottom-0 left-1/2 -ml-px w-0.5
                          bg-emerald-600 group-hover:bg-emerald-500" />
          {/* arrow ">" pointing right (into kept region) */}
          <div
            className="absolute left-1/2 top-1/2 -translate-y-1/2
                       w-0 h-0 border-y-[6px] border-y-transparent
                       border-l-[7px] border-l-emerald-600
                       group-hover:border-l-emerald-500"
          />
        </div>

        {/* trim end handle: thin vertical bar at boundary, "<" pointing into kept region */}
        <div
          onPointerDown={startDrag('end', 'ew-resize')}
          className="absolute top-0 bottom-0 w-4 -ml-2 z-10 cursor-ew-resize
                     touch-none flex items-center group"
          style={{ left: pct(trimEnd) }}
          title={`trim end ${fmt(trimEnd)}s`}
        >
          <div className="absolute top-0 bottom-0 left-1/2 -ml-px w-0.5
                          bg-emerald-600 group-hover:bg-emerald-500" />
          <div
            className="absolute right-1/2 top-1/2 -translate-y-1/2
                       w-0 h-0 border-y-[6px] border-y-transparent
                       border-r-[7px] border-r-emerald-600
                       group-hover:border-r-emerald-500"
          />
        </div>

        {/* playhead (thin vertical bar with wider invisible hit area) */}
        <div
          onPointerDown={startDrag('playhead', 'grabbing')}
          className="absolute top-0 bottom-0 w-3 -ml-1.5 z-20 touch-none
                     cursor-grab active:cursor-grabbing group"
          style={{ left: pct(playhead) }}
          title={`playhead ${fmt(playhead)}s`}
        >
          <div className="absolute top-0 bottom-0 left-1/2 -ml-px w-0.5
                          bg-red-500 group-hover:bg-red-400" />
        </div>
      </div>
    </div>
  )
}
