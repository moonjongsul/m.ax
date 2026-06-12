import * as ROSLIB from 'roslib'
import { useEffect } from 'react'
import { useSelector } from 'react-redux'
import { getRos } from './useRosConnection'

export function useRosTopicSubscription(name, type, callback, cell = 'kitting') {
  const connected = useSelector((s) => (s.ros.cells[cell] || {}).connected)

  useEffect(() => {
    if (!connected) { return }
    const ros = getRos(cell)
    if (!ros) { return }
    const topic = new ROSLIB.Topic({ ros, name, messageType: type })
    topic.subscribe(callback)
    return () => { try { topic.unsubscribe() } catch { /* noop */ } }
  }, [connected, name, type, callback, cell])
}
