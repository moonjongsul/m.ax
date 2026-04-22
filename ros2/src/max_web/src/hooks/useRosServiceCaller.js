import ROSLIB from 'roslib'
import { useCallback } from 'react'
import { getRos } from './useRosConnection'

export function useRosServiceCaller() {
  const call = useCallback((name, type, payload, timeoutMs = 120000) => {
    return new Promise((resolve, reject) => {
      const ros = getRos()
      if (!ros) { reject(new Error('ROS not connected')); return }
      const svc = new ROSLIB.Service({ ros, name, serviceType: type })
      const req = new ROSLIB.ServiceRequest(payload || {})
      const timer = setTimeout(() => reject(new Error(`service timeout: ${name}`)), timeoutMs)
      svc.callService(req, (res) => {
        clearTimeout(timer)
        resolve(res)
      }, (err) => {
        clearTimeout(timer)
        reject(new Error(String(err)))
      })
    })
  }, [])

  return { call }
}
