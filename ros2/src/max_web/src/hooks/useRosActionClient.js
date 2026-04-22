import ROSLIB from 'roslib'
import { useCallback, useRef } from 'react'
import { getRos } from './useRosConnection'

/**
 * Minimal Action client wrapper using roslibjs.
 * roslibjs ActionClient is rosbridge-based; returns a handle with cancel() and event callbacks.
 */
export function useRosActionClient(serverName, actionType) {
  const clientRef = useRef(null)
  const goalRef = useRef(null)

  const ensureClient = () => {
    const ros = getRos()
    if (!ros) { throw new Error('ROS not connected') }
    if (!clientRef.current) {
      clientRef.current = new ROSLIB.ActionClient({
        ros,
        serverName,
        actionName: actionType,
      })
    }
    return clientRef.current
  }

  const sendGoal = useCallback((goalMsg, { onFeedback, onResult } = {}) => {
    const client = ensureClient()
    const goal = new ROSLIB.Goal({ actionClient: client, goalMessage: goalMsg })
    if (onFeedback) { goal.on('feedback', onFeedback) }
    if (onResult) { goal.on('result', onResult) }
    goal.send()
    goalRef.current = goal
    return goal
  }, [serverName, actionType])

  const cancel = useCallback(() => {
    if (goalRef.current) {
      try { goalRef.current.cancel() } catch { /* noop */ }
    }
  }, [])

  return { sendGoal, cancel }
}
