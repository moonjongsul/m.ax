import { useState } from 'react'
import { useSelector } from 'react-redux'
import { useRosServiceCaller } from '../hooks/useRosServiceCaller'

// Picking-cell services called directly on the picking PC (cell='picking').
// /picking_cell/pick is std_srvs/Trigger (empty request).
export default function PickingControl() {
  const { call } = useRosServiceCaller('picking')
  const connected = useSelector((s) => (s.ros.cells.picking || {}).connected)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState(null)

  const onPick = async () => {
    setBusy(true)
    setResult(null)
    try {
      // A full pick cycle (move → detect → descend → grip → place → return)
      // takes well over rosbridge's 5s default, so raise the bridge-side
      // service-call timeout. 120000ms is the outer browser-side bound.
      const res = await call(
        '/picking_cell/pick', 'std_srvs/srv/Trigger', {}, 120000, 20,
      )
      setResult(res)
    } catch (e) {
      setResult({ success: false, message: String(e.message || e) })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="bg-white rounded-lg shadow p-4 space-y-3">
      <h3 className="text-lg font-semibold">Picking Control</h3>
      {!connected && (
        <p className="text-xs text-amber-600">Picking cell disconnected.</p>
      )}
      <button
        onClick={onPick}
        disabled={!connected || busy}
        className="px-4 py-2 rounded bg-blue-600 text-white text-sm hover:bg-blue-700 disabled:bg-gray-300 disabled:text-gray-500"
      >
        {busy ? 'Picking…' : 'Start Pick Cycle'}
      </button>
      <p className="text-xs text-gray-500">
        Calls /picking_cell/pick (std_srvs/Trigger)
      </p>
      {result && (
        <p className={`text-sm ${result.success ? 'text-green-600' : 'text-red-600'}`}>
          {result.message}
        </p>
      )}
    </div>
  )
}
