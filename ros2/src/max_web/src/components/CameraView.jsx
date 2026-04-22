import { useSelector } from 'react-redux'
import { selectVideoBaseUrl } from '../features/ros/rosSlice'

const CAMERA_TOPICS = [
  { name: 'wrist_front', topic: '/wrist/front/color/image_raw' },
  { name: 'wrist_rear',  topic: '/wrist/rear/color/image_raw' },
  { name: 'front_view',  topic: '/front_view/color/image_raw' },
  { name: 'side_view',   topic: '/side_view/color/image_raw' },
]

export default function CameraView() {
  const base = useSelector(selectVideoBaseUrl)
  return (
    <div className="grid grid-cols-2 gap-2">
      {CAMERA_TOPICS.map(({ name, topic }) => (
        <div key={name} className="bg-black rounded-lg overflow-hidden">
          <div className="px-2 py-1 text-xs text-gray-300 bg-gray-800">{name}</div>
          <img
            src={`${base}/stream?topic=${topic}&type=mjpeg&quality=60`}
            alt={name}
            className="w-full h-56 object-contain bg-black"
          />
        </div>
      ))}
    </div>
  )
}
