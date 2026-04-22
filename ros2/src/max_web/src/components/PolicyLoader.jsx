import { useState } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { setFormField } from '../features/inference/inferenceSlice'
import { useRosServiceCaller } from '../hooks/useRosServiceCaller'

function Field({ label, value, onChange, disabled, placeholder }) {
  return (
    <label className="block text-sm">
      <span className="text-gray-600">{label}</span>
      <input
        type="text"
        className="mt-1 block w-full rounded border border-gray-300 px-2 py-1 disabled:bg-gray-100"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        placeholder={placeholder}
      />
    </label>
  )
}

export default function PolicyLoader() {
  const dispatch = useDispatch()
  const form = useSelector((s) => s.inference.form)
  const serverState = useSelector((s) => s.inference.serverState)
  const { call } = useRosServiceCaller()
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState(null)

  const disabled = busy || serverState === 'loading' || serverState === 'running'
  const set = (key) => (value) => dispatch(setFormField({ key, value }))

  const onLoad = async () => {
    setBusy(true)
    setResult(null)
    try {
      const res = await call('/inference/load_policy', 'max_interfaces/srv/LoadPolicy', {
        framework: form.framework,
        policy: form.policy,
        checkpoint: form.checkpoint,
        device: form.device,
        dtype: form.dtype,
      })
      setResult(res)
    } catch (e) {
      setResult({ success: false, message: String(e.message || e) })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="bg-white rounded-lg shadow p-4 space-y-3">
      <h3 className="text-lg font-semibold">Load Policy</h3>
      <Field label="Framework" value={form.framework} onChange={set('framework')} disabled={disabled} placeholder="lerobot" />
      <Field label="Policy"    value={form.policy}    onChange={set('policy')}    disabled={disabled} placeholder="pi0.5 | smolvla" />
      <Field label="Checkpoint" value={form.checkpoint} onChange={set('checkpoint')} disabled={disabled} placeholder="/path or HF repo" />
      <Field label="Device"    value={form.device}    onChange={set('device')}    disabled={disabled} placeholder="cuda | cpu" />
      <Field label="dtype"     value={form.dtype}     onChange={set('dtype')}     disabled={disabled} placeholder="bfloat16 | float32" />
      <button
        onClick={onLoad}
        disabled={disabled}
        className="w-full rounded bg-blue-600 text-white py-2 disabled:bg-gray-300"
      >
        {busy ? 'Loading…' : 'Load'}
      </button>
      {result && (
        <p className={`text-sm ${result.success ? 'text-green-600' : 'text-red-600'}`}>
          {result.message}
        </p>
      )}
    </div>
  )
}
