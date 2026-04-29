import { useState } from 'react'
import { useSelector } from 'react-redux'
import { useRosConnection } from './hooks/useRosConnection'
import InferencePage from './pages/InferencePage'
import LeRobotEditorPage from './pages/LeRobotEditorPage'

function ConnectionBar({ page, setPage }) {
  const { host, connected, connecting, error } = useSelector((s) => s.ros)
  const dotClass = connected ? 'bg-green-500' : connecting ? 'bg-yellow-400' : 'bg-red-500'
  return (
    <header className="flex items-center gap-3 px-4 py-2 bg-gray-900 text-gray-100 text-sm">
      <h1 className="font-semibold text-base">M.AX</h1>
      <span className={`inline-block w-2.5 h-2.5 rounded-full ${dotClass}`} />
      <span>{connected ? 'connected' : connecting ? 'connecting…' : 'disconnected'}</span>
      <span className="text-gray-400">@ {host}:9090</span>
      {error && <span className="text-red-400 ml-4">{error}</span>}
      <nav className="ml-auto flex gap-1">
        <PageTab name="inference" label="Inference" page={page} setPage={setPage} />
        <PageTab name="lerobot" label="LeRobot Editor" page={page} setPage={setPage} />
      </nav>
    </header>
  )
}

function PageTab({ name, label, page, setPage }) {
  const active = page === name
  return (
    <button
      onClick={() => setPage(name)}
      className={`px-3 py-1 rounded text-sm transition
                  ${active ? 'bg-gray-100 text-gray-900'
                           : 'text-gray-300 hover:bg-gray-800'}`}
    >
      {label}
    </button>
  )
}

export default function App() {
  useRosConnection()
  const [page, setPage] = useState('inference')
  return (
    <div className="min-h-full flex flex-col">
      <ConnectionBar page={page} setPage={setPage} />
      <main className="flex-1">
        {page === 'inference' && <InferencePage />}
        {page === 'lerobot' && <LeRobotEditorPage />}
      </main>
    </div>
  )
}
