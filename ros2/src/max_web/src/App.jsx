import { useSelector } from 'react-redux'
import { useRosConnection } from './hooks/useRosConnection'
import InferencePage from './pages/InferencePage'

function ConnectionBar() {
  const { host, connected, connecting, error } = useSelector((s) => s.ros)
  const dotClass = connected ? 'bg-green-500' : connecting ? 'bg-yellow-400' : 'bg-red-500'
  return (
    <header className="flex items-center gap-3 px-4 py-2 bg-gray-900 text-gray-100 text-sm">
      <h1 className="font-semibold text-base">M.AX</h1>
      <span className={`inline-block w-2.5 h-2.5 rounded-full ${dotClass}`} />
      <span>{connected ? 'connected' : connecting ? 'connecting…' : 'disconnected'}</span>
      <span className="text-gray-400">@ {host}:9090</span>
      {error && <span className="text-red-400 ml-4">{error}</span>}
    </header>
  )
}

export default function App() {
  useRosConnection()
  return (
    <div className="min-h-full flex flex-col">
      <ConnectionBar />
      <main className="flex-1">
        <InferencePage />
      </main>
    </div>
  )
}
