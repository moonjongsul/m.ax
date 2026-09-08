import PickingCameraView from '../components/PickingCameraView'
import PickingControl from '../components/PickingControl'
import PickingHostInput from '../components/PickingHostInput'

// The picking cell runs on a separate PC. The browser connects to it directly
// (cell='picking') for service calls and subscribes to its vision result via
// the picking PC's web_video_server. max_server is not in this path.
export default function PickingPage() {
  return (
    <div className="grid grid-cols-3 gap-4 p-4">
      <div className="col-span-2 space-y-4">
        <PickingCameraView />
      </div>
      <div className="col-span-1 space-y-4">
        <PickingHostInput />
        <PickingControl />
      </div>
    </div>
  )
}
