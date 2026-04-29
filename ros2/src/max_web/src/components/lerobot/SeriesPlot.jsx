import { useEffect, useRef } from 'react'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'

const COLORS = [
  '#2563eb', '#dc2626', '#16a34a', '#d97706',
  '#7c3aed', '#0891b2', '#be185d', '#525252',
]

/**
 * Multi-series uPlot wrapper.
 *
 * @param {object} props
 * @param {number[][]} props.data       length-N array of length-D rows
 * @param {string[]}   props.names      D channel names
 * @param {number}     props.fps
 * @param {number}     props.playhead   in seconds
 * @param {string}     props.title
 */
export default function SeriesPlot({ data, names, fps, playhead, title }) {
  const containerRef = useRef(null)
  const plotRef = useRef(null)

  // Build uPlot data exactly once per (data, names) change.
  useEffect(() => {
    if (!containerRef.current || !data || !data.length || !names?.length) return
    const N = data.length
    const D = names.length
    const xs = new Float64Array(N)
    for (let i = 0; i < N; i++) xs[i] = i / fps
    const ys = Array.from({ length: D }, () => new Float64Array(N))
    for (let i = 0; i < N; i++) {
      const row = data[i]
      for (let d = 0; d < D; d++) ys[d][i] = row[d]
    }

    const series = [
      {},
      ...names.map((n, d) => ({
        label: n,
        stroke: COLORS[d % COLORS.length],
        width: 1.25,
      })),
    ]

    const opts = {
      title,
      width: containerRef.current.clientWidth,
      height: 160,
      cursor: { drag: { x: false, y: false }, sync: { key: 'lerobot' } },
      legend: { show: true, live: false },
      scales: { x: { time: false } },
      axes: [
        { stroke: '#666', grid: { stroke: '#eee' } },
        { stroke: '#666', grid: { stroke: '#eee' } },
      ],
      series,
      hooks: {
        drawClear: [
          (u) => {
            // Vertical playhead line.
            const ctx = u.ctx
            const t = plotRef.current?._playhead
            if (typeof t !== 'number') return
            const xPos = u.valToPos(t, 'x', true)
            ctx.save()
            ctx.strokeStyle = 'rgba(220, 38, 38, 0.9)'
            ctx.lineWidth = 1.5
            ctx.beginPath()
            ctx.moveTo(xPos, u.bbox.top)
            ctx.lineTo(xPos, u.bbox.top + u.bbox.height)
            ctx.stroke()
            ctx.restore()
          },
        ],
      },
    }

    if (plotRef.current) plotRef.current.destroy()
    plotRef.current = new uPlot(opts, [xs, ...ys], containerRef.current)

    const onResize = () => {
      if (!containerRef.current || !plotRef.current) return
      plotRef.current.setSize({
        width: containerRef.current.clientWidth,
        height: 160,
      })
    }
    window.addEventListener('resize', onResize)
    return () => {
      window.removeEventListener('resize', onResize)
      if (plotRef.current) {
        plotRef.current.destroy()
        plotRef.current = null
      }
    }
  }, [data, names, fps, title])

  // Update playhead without rebuilding the plot.
  useEffect(() => {
    if (!plotRef.current) return
    plotRef.current._playhead = playhead
    plotRef.current.redraw(false, false)
  }, [playhead])

  return <div ref={containerRef} className="w-full" />
}
