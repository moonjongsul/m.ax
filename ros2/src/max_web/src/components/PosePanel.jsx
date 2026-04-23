import { useState } from 'react'
import { useSelector } from 'react-redux'
import { useRosServiceCaller } from '../hooks/useRosServiceCaller'

function PoseRow({ label, poseLocations, target, expressionType, disabled, onResult }) {
  const { call } = useRosServiceCaller()
  const [busyPoseLocation, setBusyPoseLocation] = useState(null)

  const onClick = async (poseLocation) => {
    setBusyPoseLocation(poseLocation)
    try {
      const res = await call(
        '/control/move_to_pose',
        'max_interfaces/srv/MoveToPose',
        { target, pose_location: poseLocation, expression_type: expressionType },
      )
      onResult?.(res)
    } catch (e) {
      onResult?.({ success: false, message: String(e.message || e) })
    } finally {
      setBusyPoseLocation(null)
    }
  }

  return (
    <div>
      <div className="text-sm text-gray-600 mb-1">{label}</div>
      {poseLocations.length === 0 ? (
        <p className="text-xs text-gray-400 italic">No presets defined in YAML</p>
      ) : (
        <div className="flex flex-wrap gap-2">
          {poseLocations.map((poseLocation) => (
            <button
              key={poseLocation}
              onClick={() => onClick(poseLocation)}
              disabled={disabled || busyPoseLocation !== null}
              className="px-3 py-1.5 rounded border border-gray-300 bg-gray-50 hover:bg-gray-200 text-sm disabled:bg-gray-100 disabled:text-gray-400"
            >
              {busyPoseLocation === poseLocation ? '…' : poseLocation}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

export default function PosePanel() {
  const robotPoseNames = useSelector((s) => s.inference.robotPoseNames)
  const gripperPoseNames = useSelector((s) => s.inference.gripperPoseNames)
  const serverState = useSelector((s) => s.inference.serverState)
  const expressionType = useSelector((s) => s.inference.form.expressionType) || 'joint'

  const disabled = serverState === 'running' || serverState === 'loading'
  const [result, setResult] = useState(null)

  return (
    <div className="bg-white rounded-lg shadow p-4 space-y-3">
      <h3 className="text-lg font-semibold">Pose Control</h3>
      {disabled && (
        <p className="text-xs text-amber-600">
          Disabled while {serverState}.
        </p>
      )}
      <div className="text-xs text-gray-500">
        Publish via {expressionType === 'joint' ? 'joint_state' : 'goal_pose (quat)'}
      </div>

      <PoseRow
        label="Robot"
        poseLocations={robotPoseNames}
        target="robot"
        expressionType={expressionType}
        disabled={disabled}
        onResult={setResult}
      />
      <PoseRow
        label="Gripper"
        poseLocations={gripperPoseNames}
        target="gripper"
        expressionType={expressionType}
        disabled={disabled}
        onResult={setResult}
      />

      {result && (
        <p className={`text-sm ${result.success ? 'text-green-600' : 'text-red-600'}`}>
          {result.message}
        </p>
      )}
    </div>
  )
}
