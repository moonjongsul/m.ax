import { useCallback, useEffect, useRef, useState } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { useRosTopicSubscription } from '../hooks/useRosTopicSubscription'
import { setRobotStates } from '../features/telemetry/telemetrySlice'

const BUFFER_SECONDS = 10
const NOMINAL_HZ = 30
const BUFFER_SIZE = BUFFER_SECONDS * NOMINAL_HZ
const REDUX_THROTTLE_MS = 100 // 10 Hz numeric update
const CHART_DRAW_MS = 50      // 20 Hz chart redraw

const SERIES_COLORS = [
  '#ef4444', '#f97316', '#eab308', '#22c55e',
  '#14b8a6', '#3b82f6', '#8b5cf6', '#ec4899',
]

function formatNumber(v, digits = 4) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  return Number(v).toFixed(digits)
}

const POSE_LABELS = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']

function NumericSection() {
  const t = useSelector((s) => s.telemetry)
  const hasJoints = t.jointPositions.length > 0

  // "Command" column mirrors whichever action the server is currently
  // publishing: joint for representation_type=joint, pose (xyz+quat) for quat/rot6d.
  // The server sets exactly one of these valid flags per tick, so we pick
  // whichever is live.
  const cmdMode =
    t.goalPose ? 'pose'
    : t.jointCommand.length > 0 ? 'joint'
    : null

  return (
    <div className="grid grid-cols-3 gap-3 text-xs">
      <div>
        <div className="font-semibold text-gray-700 mb-1">Joint Position</div>
        {hasJoints ? (
          <table className="w-full">
            <tbody>
              {t.jointPositions.map((v, i) => (
                <tr key={i} className="border-b border-gray-100">
                  <td className="text-gray-500">{t.jointNames[i] || `j${i}`}</td>
                  <td className="text-right font-mono">{formatNumber(v)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="text-gray-400 italic">no data</p>
        )}
      </div>

      <div>
        <div className="font-semibold text-gray-700 mb-1">Current Pose (quat)</div>
        {t.currentPose ? (
          <table className="w-full font-mono">
            <tbody>
              {POSE_LABELS.map((k) => (
                <tr key={k}>
                  <td className="text-gray-500">{k}</td>
                  <td className="text-right">{formatNumber(t.currentPose[k])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="text-gray-400 italic">no data</p>
        )}
      </div>

      <div>
        <div className="font-semibold text-gray-700 mb-1">
          Command {cmdMode === 'pose' ? '(xyz + quat)' : cmdMode === 'joint' ? '(joint)' : ''}
        </div>
        {cmdMode === 'joint' ? (
          <table className="w-full font-mono text-blue-600">
            <tbody>
              {t.jointCommand.map((v, i) => (
                <tr key={i} className="border-b border-gray-100">
                  <td className="text-gray-500">{t.jointNames[i] || `j${i}`}</td>
                  <td className="text-right">{formatNumber(v)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : cmdMode === 'pose' ? (
          <table className="w-full font-mono text-blue-600">
            <tbody>
              {POSE_LABELS.map((k) => (
                <tr key={k}>
                  <td className="text-gray-500">{k}</td>
                  <td className="text-right">{formatNumber(t.goalPose[k])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="text-gray-400 italic">no data</p>
        )}

        <div className="font-semibold text-gray-700 mt-3 mb-1">Gripper</div>
        <table className="w-full font-mono">
          <tbody>
            <tr>
              <td className="text-gray-500">state</td>
              <td className="text-right">{formatNumber(t.gripperState)}</td>
            </tr>
            <tr>
              <td className="text-gray-500">cmd</td>
              <td className="text-right text-blue-600">{formatNumber(t.gripperCommand)}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  )
}

function useRingBuffer(numSeries) {
  const ref = useRef(null)
  if (ref.current === null || ref.current.series.length !== numSeries) {
    ref.current = {
      t: new Array(BUFFER_SIZE).fill(NaN),
      series: Array.from({ length: numSeries }, () => new Array(BUFFER_SIZE).fill(NaN)),
      idx: 0,
      filled: 0,
    }
  }
  return ref
}

function push(buf, t, values) {
  const i = buf.idx
  buf.t[i] = t
  for (let k = 0; k < buf.series.length; k++) {
    buf.series[k][i] = values[k] === undefined ? NaN : values[k]
  }
  buf.idx = (i + 1) % BUFFER_SIZE
  buf.filled = Math.min(buf.filled + 1, BUFFER_SIZE)
}

// Returns [timestamps, ...seriesArrays] aligned chronologically.
function snapshot(buf) {
  const n = buf.filled
  if (n === 0) {
    return [new Array(0), ...buf.series.map(() => new Array(0))]
  }
  const start = buf.filled < BUFFER_SIZE ? 0 : buf.idx
  const out = [new Array(n)]
  for (let k = 0; k < buf.series.length; k++) out.push(new Array(n))
  for (let j = 0; j < n; j++) {
    const src = (start + j) % BUFFER_SIZE
    out[0][j] = buf.t[src]
    for (let k = 0; k < buf.series.length; k++) {
      out[k + 1][j] = buf.series[k][src]
    }
  }
  return out
}

function Chart({ title, bufRef, labels, height = 160 }) {
  const wrapRef = useRef(null)
  const plotRef = useRef(null)

  useEffect(() => {
    if (!wrapRef.current) return
    const series = [
      {},
      ...labels.map((label, i) => ({
        label,
        stroke: SERIES_COLORS[i % SERIES_COLORS.length],
        width: 1.25,
        spanGaps: true,
      })),
    ]
    const opts = {
      width: wrapRef.current.clientWidth,
      height,
      cursor: { drag: { x: false, y: false } },
      legend: { show: true, live: false },
      scales: { x: { time: true } },
      series,
      axes: [
        { stroke: '#6b7280', grid: { stroke: '#e5e7eb' } },
        { stroke: '#6b7280', grid: { stroke: '#e5e7eb' } },
      ],
    }
    const init = snapshot(bufRef.current)
    if (init[0].length === 0) {
      init[0] = [Date.now() / 1000]
      for (let k = 1; k < init.length; k++) init[k] = [NaN]
    }
    plotRef.current = new uPlot(opts, init, wrapRef.current)

    const onResize = () => {
      if (!wrapRef.current || !plotRef.current) return
      plotRef.current.setSize({ width: wrapRef.current.clientWidth, height })
    }
    window.addEventListener('resize', onResize)

    const interval = setInterval(() => {
      if (!plotRef.current) return
      const data = snapshot(bufRef.current)
      if (data[0].length > 0) plotRef.current.setData(data)
    }, CHART_DRAW_MS)

    return () => {
      clearInterval(interval)
      window.removeEventListener('resize', onResize)
      plotRef.current?.destroy()
      plotRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [labels.length, height])

  return (
    <div>
      <div className="text-xs font-semibold text-gray-700 mb-1">{title}</div>
      <div ref={wrapRef} className="w-full" />
    </div>
  )
}

export default function RobotStatesPanel() {
  const dispatch = useDispatch()
  const [jointCount, setJointCount] = useState(0)
  // 'joint' | 'pose' | null — picked from valid flags, sticks across ticks
  // where neither is valid so the chart doesn't churn its series count.
  const [cmdMode, setCmdMode] = useState(null)

  const jointStateBuf = useRingBuffer(Math.max(jointCount, 1))
  const poseBuf = useRingBuffer(7)        // x, y, z, qx, qy, qz, qw
  const gripperBuf = useRingBuffer(2)     // state, cmd
  const cmdJointBuf = useRingBuffer(Math.max(jointCount, 1))
  const cmdPoseBuf = useRingBuffer(7)

  const lastDispatchRef = useRef(0)

  const onMsg = useCallback((msg) => {
    const tSec = (msg.header?.stamp?.sec ?? 0) + (msg.header?.stamp?.nanosec ?? 0) / 1e9
    const t = tSec > 0 ? tSec : Date.now() / 1000

    const positions = msg.joint_positions || []
    if (positions.length !== jointCount) {
      setJointCount(positions.length)
    }

    if (msg.goal_pose_valid) {
      if (cmdMode !== 'pose') setCmdMode('pose')
    } else if (msg.joint_command_valid) {
      if (cmdMode !== 'joint') setCmdMode('joint')
    }

    push(jointStateBuf.current, t, positions)
    push(poseBuf.current, t, msg.current_pose_valid
      ? [
          msg.current_pose_x, msg.current_pose_y, msg.current_pose_z,
          msg.current_pose_qx, msg.current_pose_qy,
          msg.current_pose_qz, msg.current_pose_qw,
        ]
      : [NaN, NaN, NaN, NaN, NaN, NaN, NaN])
    push(gripperBuf.current, t, [
      msg.gripper_state_valid ? msg.gripper_state : NaN,
      msg.gripper_command_valid ? msg.gripper_command : NaN,
    ])

    push(
      cmdJointBuf.current, t,
      msg.joint_command_valid ? (msg.joint_command || []) : positions.map(() => NaN),
    )
    push(
      cmdPoseBuf.current, t,
      msg.goal_pose_valid
        ? [
            msg.goal_pose_x, msg.goal_pose_y, msg.goal_pose_z,
            msg.goal_pose_qx, msg.goal_pose_qy,
            msg.goal_pose_qz, msg.goal_pose_qw,
          ]
        : [NaN, NaN, NaN, NaN, NaN, NaN, NaN],
    )

    const now = Date.now()
    if (now - lastDispatchRef.current >= REDUX_THROTTLE_MS) {
      lastDispatchRef.current = now
      dispatch(setRobotStates(msg))
    }
  }, [dispatch, jointCount, cmdMode, jointStateBuf, poseBuf, gripperBuf, cmdJointBuf, cmdPoseBuf])

  useRosTopicSubscription(
    '/control/robot_state_package',
    'max_interfaces/msg/RobotStatesPackage',
    onMsg,
  )

  const jointLabels = Array.from({ length: Math.max(jointCount, 1) }, (_, i) => `j${i}`)

  return (
    <div className="bg-white rounded-lg shadow p-4 space-y-4">
      <h3 className="text-lg font-semibold">Robot States</h3>
      <NumericSection />
      <div className="grid grid-cols-2 gap-4">
        <Chart title="Joint Position (state)" bufRef={jointStateBuf} labels={jointLabels} />
        {cmdMode === 'pose' ? (
          <Chart title="Command (xyz + quat)" bufRef={cmdPoseBuf} labels={POSE_LABELS} />
        ) : (
          <Chart title="Command (joint)" bufRef={cmdJointBuf} labels={jointLabels} />
        )}
        <Chart title="Current Pose (xyz + quat)" bufRef={poseBuf} labels={POSE_LABELS} />
        <Chart title="Gripper (state, cmd)" bufRef={gripperBuf} labels={['state', 'cmd']} />
      </div>
    </div>
  )
}
