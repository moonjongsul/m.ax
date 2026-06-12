import { useState } from 'react'
import { useSelector } from 'react-redux'
import { useRosConnection } from './hooks/useRosConnection'
import InferencePage from './pages/InferencePage'
import LeRobotEditorPage from './pages/LeRobotEditorPage'
import PickingPage from './pages/PickingPage'

function CellStatus({ label, cell }) {
  const c = useSelector((s) => s.ros.cells[cell]) || {}
  const { host, connected, connecting, error } = c
  const dotClass = connected ? 'bg-green-500' : connecting ? 'bg-yellow-400' : 'bg-red-500'
  return (
    <span className="flex items-center gap-1.5">
      <span className="text-gray-400 text-xs">{label}</span>
      <span className={`inline-block w-2.5 h-2.5 rounded-full ${dotClass}`} />
      <span className="text-gray-400 text-xs">@ {host}:9090</span>
      {error && <span className="text-red-400 text-xs">{error}</span>}
    </span>
  )
}

function ConnectionBar({ page, setPage }) {
  return (
    <header className="flex items-center gap-4 px-4 py-2 bg-gray-900 text-gray-100 text-sm">
      <h1 className="font-semibold text-base">M.AX</h1>
      <CellStatus label="kitting" cell="kitting" />
      <CellStatus label="picking" cell="picking" />
      <nav className="ml-auto flex gap-1">
        <PageTab name="inference" label="Inference" page={page} setPage={setPage} />
        <PageTab name="picking" label="Picking" page={page} setPage={setPage} />
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
  const [page, setPage] = useState('inference')
  // Keep both cells connected for the whole app lifetime so kitting and
  // picking run simultaneously regardless of the active page.
  useRosConnection('kitting')
  useRosConnection('picking')
  return (
    <div className="min-h-full flex flex-col">
      <ConnectionBar page={page} setPage={setPage} />
      <main className="flex-1">
        {page === 'inference' && <InferencePage />}
        {page === 'picking' && <PickingPage />}
        {page === 'lerobot' && <LeRobotEditorPage />}
      </main>
    </div>
  )
}
