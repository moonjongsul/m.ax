import { useEffect, useRef } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { setPlayhead } from '../../features/lerobot/lerobotSlice'

function shortKey(k) {
  // observation.images.front_view -> front_view
  return k.replace(/^observation\.images\./, '')
}

/**
 * One <video> per video_key. The element's src points at the (possibly
 * multi-episode) chunk file; we constrain playback to the episode's
 * [from_timestamp, to_timestamp] window. The Redux `playhead` is
 * episode-relative seconds and drives currentTime on each video.
 */
export default function VideoGrid({ primaryRef }) {
  const dispatch = useDispatch()
  const frames = useSelector((s) => s.lerobot.frames)
  const playhead = useSelector((s) => s.lerobot.playhead)
  const playing = useSelector((s) => s.lerobot.playing)
  const playbackRate = useSelector((s) => s.lerobot.playbackRate)

  const videoRefs = useRef({}) // key -> HTMLVideoElement
  // Stamp bumped on every episode switch so timeupdate fired from stale
  // metadata loads (which can fire `t = 0 - from_timestamp` < 0 and incorrectly
  // clamp the playhead to the episode end) is ignored briefly.
  const epSwitchAtRef = useRef(0)

  // Reset src when episode changes. Pause to avoid runaway play() from prev ep.
  useEffect(() => {
    if (!frames) return
    epSwitchAtRef.current = performance.now()
    Object.entries(frames.videos).forEach(([key, meta]) => {
      const el = videoRefs.current[key]
      if (!el) return
      el.pause()
      // Force src reload; lerobot mp4s contain multiple episodes back-to-back.
      const newSrc = meta.url
      if (el.dataset.currentUrl !== newSrc) {
        el.src = newSrc
        el.dataset.currentUrl = newSrc
        el.load()
      }
    })
  }, [frames])

  // Sync currentTime to playhead. We only push when the diff is large enough
  // to avoid fighting the browser when it's playing naturally.
  useEffect(() => {
    if (!frames) return
    Object.entries(frames.videos).forEach(([key, meta]) => {
      const el = videoRefs.current[key]
      if (!el || !isFinite(el.duration)) return
      const target = meta.from_timestamp + playhead
      if (Math.abs(el.currentTime - target) > 0.08) {
        try { el.currentTime = target } catch { /* not yet seekable */ }
      }
    })
  }, [playhead, frames])

  // Play / pause
  useEffect(() => {
    if (!frames) return
    Object.values(videoRefs.current).forEach((el) => {
      if (!el) return
      if (playing) el.play().catch(() => {})
      else el.pause()
    })
  }, [playing, frames])

  // Playback rate
  useEffect(() => {
    Object.values(videoRefs.current).forEach((el) => {
      if (el) el.playbackRate = playbackRate
    })
  }, [playbackRate, frames])

  // Drive Redux playhead from the *primary* video's timeupdate so plots
  // stay synced while playing. Other videos follow via the effect above.
  const onPrimaryTimeUpdate = (key, meta) => () => {
    const el = videoRefs.current[key]
    if (!el || !frames) return
    // Ignore timeupdate events that fire right after switching episodes —
    // the video element is still seeking from the previous episode's
    // footage (multiple episodes share one mp4) so currentTime can briefly
    // be < from_timestamp or > to_timestamp.
    if (performance.now() - epSwitchAtRef.current < 400) return
    const t = el.currentTime - meta.from_timestamp
    const dur = frames.length / frames.fps
    if (t < 0) {
      // Below this episode's window; clamp upward without touching playhead.
      try { el.currentTime = meta.from_timestamp } catch { /* not seekable */ }
      return
    }
    if (t > dur + 0.05) {
      // Crossed into the next episode's footage while playing — pause at end.
      el.pause()
      try { el.currentTime = meta.to_timestamp } catch { /* not seekable */ }
      dispatch(setPlayhead(dur))
      return
    }
    dispatch(setPlayhead(t))
  }

  if (!frames) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-400">
        Select an episode to view videos.
      </div>
    )
  }

  const keys = Object.keys(frames.videos)
  // Pick the first key as the timeupdate driver.
  const primaryKey = keys[0]

  return (
    <div
      className="grid gap-2 p-2"
      style={{ gridTemplateColumns: `repeat(${keys.length}, minmax(0, 1fr))` }}
    >
      {keys.map((key) => {
        const meta = frames.videos[key]
        const isPrimary = key === primaryKey
        return (
          <div key={key} className="bg-black rounded overflow-hidden flex flex-col">
            <video
              ref={(el) => {
                videoRefs.current[key] = el
                if (isPrimary && primaryRef) primaryRef.current = el
              }}
              className="w-full aspect-[4/3] bg-black"
              muted
              playsInline
              preload="auto"
              onTimeUpdate={isPrimary ? onPrimaryTimeUpdate(key, meta) : undefined}
            />
            <div className="px-2 py-1 text-xs text-gray-200 bg-gray-900 font-mono truncate">
              {shortKey(key)}
            </div>
          </div>
        )
      })}
    </div>
  )
}
