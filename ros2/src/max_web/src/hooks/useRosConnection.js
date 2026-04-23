import { useEffect, useRef } from 'react'
import * as ROSLIB from 'roslib'
import { useDispatch, useSelector } from 'react-redux'
import {
  selectRosbridgeUrl,
  setConnected,
  setConnecting,
  setError,
} from '../features/ros/rosSlice'

let rosSingleton = null

export function getRos() {
  return rosSingleton
}

export function useRosConnection() {
  const dispatch = useDispatch()
  const url = useSelector(selectRosbridgeUrl)
  const connected = useSelector((s) => s.ros.connected)
  const rosRef = useRef(null)

  useEffect(() => {
    dispatch(setConnecting(true))
    const ros = new ROSLIB.Ros({ url })
    rosSingleton = ros
    rosRef.current = ros

    ros.on('connection', () => {
      dispatch(setConnecting(false))
      dispatch(setConnected(true))
    })
    ros.on('close', () => {
      dispatch(setConnected(false))
      dispatch(setConnecting(false))
    })
    ros.on('error', (err) => {
      dispatch(setError(String(err?.message || err || 'connection error')))
    })

    return () => {
      try { ros.close() } catch { /* noop */ }
      if (rosSingleton === ros) { rosSingleton = null }
    }
  }, [url, dispatch])

  return { connected }
}
