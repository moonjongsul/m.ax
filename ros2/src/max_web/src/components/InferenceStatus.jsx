import { useSelector } from 'react-redux'

const STATE_COLORS = {
  idle: 'bg-gray-400',
  loading: 'bg-yellow-400',
  ready: 'bg-blue-500',
  running: 'bg-green-500',
  error: 'bg-red-500',
}

export default function InferenceStatus() {
  const { serverState, detail, policy, taskInstruction, representationType, step, lastAction } =
    useSelector((s) => s.inference)

  const dotClass = STATE_COLORS[serverState] || 'bg-gray-400'

  return (
    <div className="bg-white rounded-lg shadow p-4 space-y-3">
      <div className="flex items-center gap-2">
        <span className={`inline-block w-3 h-3 rounded-full ${dotClass}`} />
        <h3 className="text-lg font-semibold">{serverState.toUpperCase()}</h3>
      </div>
      {detail && <p className="text-sm text-red-600">{detail}</p>}

      <div className="text-sm space-y-1">
        <div><span className="text-gray-500">Framework:</span> {policy.framework || '—'}</div>
        <div><span className="text-gray-500">Policy:</span> {policy.name || '—'}</div>
        <div className="break-all"><span className="text-gray-500">Checkpoint:</span> {policy.checkpoint || '—'}</div>
        <div><span className="text-gray-500">Device / dtype:</span> {policy.device || '—'} / {policy.dtype || '—'}</div>
        <div><span className="text-gray-500">Task:</span> {taskInstruction || '—'}</div>
        <div><span className="text-gray-500">Representation:</span> {representationType || '—'}</div>
        <div><span className="text-gray-500">Step:</span> {step}</div>
        {lastAction.length > 0 && (
          <div className="break-all">
            <span className="text-gray-500">Last action:</span>{' '}
            [{lastAction.map((v) => v.toFixed(3)).join(', ')}]
          </div>
        )}
      </div>
    </div>
  )
}
