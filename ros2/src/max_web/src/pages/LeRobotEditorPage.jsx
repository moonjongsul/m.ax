import DatasetPathInput from '../components/lerobot/DatasetPathInput'
import EpisodeList from '../components/lerobot/EpisodeList'
import EpisodeViewer from '../components/lerobot/EpisodeViewer'
import TimelineBar from '../components/lerobot/TimelineBar'

export default function LeRobotEditorPage() {
  return (
    <div className="flex flex-col h-[calc(100vh-80px)]">
      <DatasetPathInput />
      <div className="flex-1 flex min-h-0">
        {/* Left: episode list */}
        <aside className="w-64 border-r border-gray-200 bg-white overflow-hidden flex flex-col">
          <EpisodeList />
        </aside>
        {/* Right: episode viewer */}
        <section className="flex-1 flex flex-col min-w-0 bg-white">
          <EpisodeViewer />
          <TimelineBar />
        </section>
      </div>
    </div>
  )
}
