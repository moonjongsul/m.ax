import { useEffect, useRef } from 'react'
import * as ROSLIB from 'roslib'
import { useDispatch, useSelector } from 'react-redux'
import {
  selectRosbridgeUrl,
  setConnected,
  setConnecting,
  setError,
} from '../features/ros/rosSlice'

// One ROSLIB.Ros instance per cell ('kitting', 'picking', ...). The browser
// can hold multiple WebSocket connections at once, so each cell talks to its
// own rosbridge independently.
const rosByCell = new Map()

export function getRos(cell = 'kitting') {
  return rosByCell.get(cell) || null
}

export function useRosConnection(cell = 'kitting') {
  const dispatch = useDispatch()
  const url = useSelector(selectRosbridgeUrl(cell))
  const connected = useSelector((s) => (s.ros.cells[cell] || {}).connected)
  const rosRef = useRef(null)

  useEffect(() => {
    dispatch(setConnecting({ cell, value: true }))
    const ros = new ROSLIB.Ros({ url })
    rosByCell.set(cell, ros)
    rosRef.current = ros

    ros.on('connection', () => {
      dispatch(setConnecting({ cell, value: false }))
      dispatch(setConnected({ cell, value: true }))
    })
    ros.on('close', () => {
      dispatch(setConnected({ cell, value: false }))
      dispatch(setConnecting({ cell, value: false }))
    })
    ros.on('error', (err) => {
      dispatch(setError({ cell, error: String(err?.message || err || 'connection error') }))
    })

    return () => {
      try { ros.close() } catch { /* noop */ }
      if (rosByCell.get(cell) === ros) { rosByCell.delete(cell) }
    }
  }, [cell, url, dispatch])

  return { connected }
}
