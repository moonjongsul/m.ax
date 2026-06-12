import * as ROSLIB from 'roslib'
import { useCallback } from 'react'
import { getRos } from './useRosConnection'

// timeoutMs        — browser-side Promise timeout (outer bound)
// bridgeTimeoutSec  — rosbridge-side service-call timeout, in seconds, forwarded
//                     to rosbridge (default there is only 5s). Pass a larger
//                     value for long-running services (e.g. a pick cycle).
export function useRosServiceCaller(cell = 'kitting') {
  const call = useCallback(
    (name, type, payload, timeoutMs = 120000, bridgeTimeoutSec = undefined) => {
      return new Promise((resolve, reject) => {
        const ros = getRos(cell)
        if (!ros) { reject(new Error('ROS not connected')); return }
        const svc = new ROSLIB.Service({ ros, name, serviceType: type })
        const req = payload || {}
        const timer = setTimeout(() => reject(new Error(`service timeout: ${name}`)), timeoutMs)
        svc.callService(req, (res) => {
          clearTimeout(timer)
          resolve(res)
        }, (err) => {
          clearTimeout(timer)
          reject(new Error(String(err)))
        }, bridgeTimeoutSec)
      })
    },
    [cell],
  )

  return { call }
}
