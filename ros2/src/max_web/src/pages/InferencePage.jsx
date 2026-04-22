import { useCallback } from 'react'
import { useDispatch } from 'react-redux'
import CameraView from '../components/CameraView'
import InferenceStatus from '../components/InferenceStatus'
import PolicyLoader from '../components/PolicyLoader'
import InferenceControl from '../components/InferenceControl'
import { useRosTopicSubscription } from '../hooks/useRosTopicSubscription'
import { setStatus } from '../features/inference/inferenceSlice'

export default function InferencePage() {
  const dispatch = useDispatch()
  const onStatus = useCallback((msg) => dispatch(setStatus(msg)), [dispatch])
  useRosTopicSubscription('/inference/status', 'max_interfaces/msg/InferenceStatus', onStatus)

  return (
    <div className="grid grid-cols-3 gap-4 p-4">
      <div className="col-span-2">
        <CameraView />
      </div>
      <div className="col-span-1 space-y-4">
        <InferenceStatus />
        <PolicyLoader />
        <InferenceControl />
      </div>
    </div>
  )
}
