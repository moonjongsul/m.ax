import { useState, useEffect } from 'react'
import { useSelector, useDispatch } from 'react-redux'
import { setHost } from '../features/ros/rosSlice'

// Lets the user point the picking cell at the separate PC's rosbridge host.
// Changing the host tears down and re-opens the picking connection
// (useRosConnection depends on the derived ws URL).
//
// The default host comes from the server (picking_cell.ip via /inference/status,
// applied in App.jsx). Until the user edits the field, mirror the store host so
// the input reflects that server-provided value once it arrives.
export default function PickingHostInput() {
  const dispatch = useDispatch()
  const cell = useSelector((s) => s.ros.cells.picking) || {}
  const [value, setValue] = useState(cell.host || '')
  const [dirty, setDirty] = useState(false)

  useEffect(() => {
    if (!dirty) { setValue(cell.host || '') }
  }, [cell.host, dirty])

  const apply = () => {
    const host = value.trim()
    if (host) {
      dispatch(setHost({ cell: 'picking', host }))
      setDirty(false)
    }
  }

  return (
    <div className="bg-white rounded-lg shadow p-4 space-y-2">
      <h3 className="text-lg font-semibold">Picking Host</h3>
      <div className="flex gap-2">
        <input
          type="text"
          value={value}
          onChange={(e) => { setValue(e.target.value); setDirty(true) }}
          onKeyDown={(e) => { if (e.key === 'Enter') { apply() } }}
          placeholder="picking PC host / IP"
          className="flex-1 px-2 py-1.5 border border-gray-300 rounded text-sm"
        />
        <button
          onClick={apply}
          className="px-3 py-1.5 rounded bg-gray-800 text-white text-sm hover:bg-gray-700"
        >
          Connect
        </button>
      </div>
      <p className="text-xs text-gray-500">
        rosbridge ws://{cell.host}:{cell.rosbridgePort}
      </p>
    </div>
  )
}
