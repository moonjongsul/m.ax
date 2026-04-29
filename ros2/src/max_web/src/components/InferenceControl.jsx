import { useRef } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { setFeedback, setFormField } from '../features/inference/inferenceSlice'
import { useRosActionClient } from '../hooks/useRosActionClient'

const COMMAND_START = 0
const COMMAND_STOP = 1

export default function InferenceControl() {
  const dispatch = useDispatch()
  const form = useSelector((s) => s.inference.form)
  const serverState = useSelector((s) => s.inference.serverState)
  const { sendGoal, cancel } = useRosActionClient('/inference/run', 'max_interfaces/action/RunInference')
  const goalRef = useRef(null)

  const canStart = serverState === 'ready'
  const canStop = serverState === 'running'

  const onStart = () => {
    goalRef.current = sendGoal(
      {
        command: COMMAND_START,
        task_instruction: form.taskInstruction,
        representation_type: form.representationType,
      },
      {
        onFeedback: (fb) => dispatch(setFeedback(fb)),
        onResult: (result) => {
          // result logged via status topic; nothing else needed here
          goalRef.current = null
        },
      },
    )
  }

  const onStop = () => {
    cancel()
  }

  return (
    <div className="bg-white rounded-lg shadow p-4 space-y-3">
      <h3 className="text-lg font-semibold">Inference Control</h3>

      <label className="block text-sm">
        <span className="text-gray-600">Task instruction</span>
        <textarea
          className="mt-1 block w-full rounded border border-gray-300 px-2 py-1 h-20 disabled:bg-gray-100"
          value={form.taskInstruction}
          onChange={(e) => dispatch(setFormField({ key: 'taskInstruction', value: e.target.value }))}
          disabled={!canStart}
        />
      </label>

      <label className="block text-sm">
        <span className="text-gray-600">Representation type</span>
        <select
          className="mt-1 block w-full rounded border border-gray-300 px-2 py-1 disabled:bg-gray-100"
          value={form.representationType}
          onChange={(e) => dispatch(setFormField({ key: 'representationType', value: e.target.value }))}
          disabled={!canStart}
        >
          <option value="joint">joint</option>
          <option value="quat">quat</option>
          <option value="rot6d">rot6d</option>
        </select>
      </label>

      <div className="grid grid-cols-2 gap-2">
        <button
          onClick={onStart}
          disabled={!canStart}
          className="rounded bg-green-600 text-white py-2 disabled:bg-gray-300"
        >
          ▶ Start
        </button>
        <button
          onClick={onStop}
          disabled={!canStop}
          className="rounded bg-red-600 text-white py-2 disabled:bg-gray-300"
        >
          ■ Stop
        </button>
      </div>
    </div>
  )
}
