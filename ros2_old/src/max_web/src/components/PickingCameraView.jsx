import { useCallback, useEffect, useState } from 'react'
import { useRosTopicSubscription } from '../hooks/useRosTopicSubscription'

// Vision detection result from the picking cell. Instead of web_video_server
// (which needs a raw image_transport topic the picking node doesn't publish),
// we subscribe to the CompressedImage topic directly over rosbridge and render
// it as a data URL. The image is single-shot / low-rate (updated per pick
// cycle), so the rosbridge traffic is negligible.
const PICKING_IMAGE_TOPIC = '/picking_cell/box_obj/detection_result/compressed'

export default function PickingCameraView() {
  const [src, setSrc] = useState(null)

  const onImage = useCallback((msg) => {
    // rosbridge delivers CompressedImage.data as a base64 string. format is
    // e.g. 'jpeg' (picking node) — map it to a valid image mime type.
    const format = (msg.format || 'jpeg').toLowerCase()
    const mime = format.includes('png') ? 'image/png' : 'image/jpeg'
    if (typeof msg.data === 'string') {
      setSrc(`data:${mime};base64,${msg.data}`)
    }
  }, [])

  useRosTopicSubscription(
    PICKING_IMAGE_TOPIC,
    'sensor_msgs/msg/CompressedImage',
    onImage,
    'picking',
  )

  return (
    <div className="grid grid-cols-1 gap-2">
      <div className="bg-black rounded-lg overflow-hidden">
        <div className="px-2 py-1 text-xs text-gray-300 bg-gray-800">
          detection_result
        </div>
        {src ? (
          <img
            src={src}
            alt="detection_result"
            className="w-full h-96 object-contain bg-black"
          />
        ) : (
          <div className="w-full h-96 flex items-center justify-center text-gray-500 text-sm">
            Waiting for detection result… (run a pick cycle)
          </div>
        )}
      </div>
    </div>
  )
}
