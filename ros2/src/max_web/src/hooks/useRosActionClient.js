import * as ROSLIB from 'roslib'
import { useCallback, useRef } from 'react'
import { getRos } from './useRosConnection'

// roslib 2.x Action API (ROS 2 native): ROSLIB.Action with sendGoal(goal, onResult, onFeedback, onFailed).
export function useRosActionClient(serverName, actionType) {
  const actionRef = useRef(null)
  const goalIdRef = useRef(null)

  const ensureAction = () => {
    const ros = getRos()
    if (!ros) { throw new Error('ROS not connected') }
    if (!actionRef.current) {
      actionRef.current = new ROSLIB.Action({
        ros,
        name: serverName,
        actionType,
      })
    }
    return actionRef.current
  }

  const sendGoal = useCallback((goalMsg, { onFeedback, onResult, onFailed } = {}) => {
    const action = ensureAction()
    const id = action.sendGoal(
      goalMsg,
      (result) => { onResult && onResult(result) },
      (feedback) => { onFeedback && onFeedback(feedback) },
      (err) => { onFailed ? onFailed(err) : console.error('[action] failed:', err) },
    )
    goalIdRef.current = id
    return id
  }, [serverName, actionType])

  const cancel = useCallback(() => {
    if (actionRef.current && goalIdRef.current) {
      try { actionRef.current.cancelGoal(goalIdRef.current) } catch { /* noop */ }
    }
  }, [])

  return { sendGoal, cancel }
}
