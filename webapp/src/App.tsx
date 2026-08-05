import { useMemo, useState } from 'react'
import { datasets } from './mockData'
import { Workbench } from './pages/Workbench'
import { Dashboard } from './pages/Dashboard'
import { Overview } from './pages/Overview'
import { Method } from './pages/Method'
import { Sparkles, LayoutDashboard, BarChart3, Compass, GitBranch } from 'lucide-react'

type Page = 'overview' | 'method' | 'workbench' | 'dashboard'

const NAV: { key: Page; label: string; icon: typeof Compass }[] = [
  { key: 'overview', label: '项目概览', icon: Compass },
  { key: 'method', label: '方法流程', icon: GitBranch },
  { key: 'workbench', label: '推荐展示', icon: LayoutDashboard },
  { key: 'dashboard', label: '效果评测', icon: BarChart3 },
]

function App() {
  const [page, setPage] = useState<Page>('overview')

  // 推荐展示：支持 GM / AO 数据集切换
  const [wbKey, setWbKey] = useState<'GM' | 'AO'>('GM')
  const dataset = useMemo(() => datasets.find((d) => d.key === wbKey) ?? datasets[0], [wbKey])
  const [userId, setUserId] = useState(dataset.users[0].id)
  const currentUser = dataset.users.find((u) => u.id === userId) ?? dataset.users[0]

  function handleRandomUser() {
    const idx = Math.floor(Math.random() * dataset.users.length)
    setUserId(dataset.users[idx].id)
  }

  function handleSwitchDataset(key: 'GM' | 'AO') {
    setWbKey(key)
    const ds = datasets.find((d) => d.key === key)
    if (ds) setUserId(ds.users[0].id)
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="sticky top-0 z-10 border-b border-gray-200 bg-white/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-3">
          <div className="flex items-center gap-8">
            <button
              onClick={() => setPage('overview')}
              className="flex items-center gap-2 transition hover:opacity-80"
            >
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-blue-600 to-cyan-500 text-white shadow-sm">
                <Sparkles size={16} />
              </div>
              <div className="flex flex-col leading-none">
                <span className="text-base font-bold text-gray-900">URLLM</span>
                <span className="text-[10px] text-gray-400">跨域序列推荐</span>
              </div>
            </button>
            <nav className="flex items-center gap-1">
              {NAV.map((item) => {
                const Icon = item.icon
                return (
                  <button
                    key={item.key}
                    onClick={() => setPage(item.key)}
                    className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition ${
                      page === item.key
                        ? 'bg-blue-50 text-blue-700'
                        : 'text-gray-500 hover:bg-gray-50 hover:text-gray-700'
                    }`}
                  >
                    <Icon size={15} />
                    {item.label}
                  </button>
                )
              })}
            </nav>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-8">
        {page === 'overview' && <Overview />}
        {page === 'method' && <Method />}
        {page === 'workbench' && (
          <Workbench
            dataset={dataset}
            currentUser={currentUser}
            onSelectUser={setUserId}
            onRandomUser={handleRandomUser}
            datasetKey={wbKey}
            onSwitchDataset={handleSwitchDataset}
          />
        )}
        {page === 'dashboard' && <Dashboard />}
      </main>
    </div>
  )
}

export default App
