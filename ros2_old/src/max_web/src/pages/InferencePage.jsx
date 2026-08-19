import CameraView from '../components/CameraView'
import InferenceStatus from '../components/InferenceStatus'
import PolicyLoader from '../components/PolicyLoader'
import InferenceControl from '../components/InferenceControl'
import PosePanel from '../components/PosePanel'
import RobotStatesPanel from '../components/RobotStatesPanel'

// /inference/status is subscribed at app level (see StatusListener in App.jsx),
// so this page just renders from the mirrored Redux state.
export default function InferencePage() {
  return (
    <div className="grid grid-cols-3 gap-4 p-4">
      <div className="col-span-2 space-y-4">
        <CameraView />
        <RobotStatesPanel />
      </div>
      <div className="col-span-1 space-y-4">
        <InferenceStatus />
        <PolicyLoader />
        <InferenceControl />
        <PosePanel />
      </div>
    </div>
  )
}
