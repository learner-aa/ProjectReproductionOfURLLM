import { useState } from 'react'
import type { Dataset, DemoUser, RecommendationResult } from '../types'
import { HistoryPanel } from '../components/HistoryPanel'
import { ResultPanel } from '../components/ResultPanel'

interface Props {
  dataset: Dataset
  currentUser: DemoUser
  onSelectUser: (id: string) => void
  onRandomUser: () => void
  datasetKey: string
  onSwitchDataset: (key: 'GM' | 'AO') => void
}

export function Workbench({ dataset, currentUser, onSelectUser, onRandomUser, datasetKey, onSwitchDataset }: Props) {
  const [result, setResult] = useState<RecommendationResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [feedback, setFeedback] = useState<'up' | 'down' | null>(null)

  function handleSelectUser(id: string) {
    onSelectUser(id)
    setResult(null)
    setFeedback(null)
  }

  function handleRandomUser() {
    onRandomUser()
    setResult(null)
    setFeedback(null)
  }

  function handleGenerate() {
    setLoading(true)
    setResult(null)
    setFeedback(null)
    window.setTimeout(() => {
      setResult(currentUser.result)
      setLoading(false)
    }, 900)
  }

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-bold text-gray-900">推荐展示</h1>
        <p className="mt-1 text-sm text-gray-500">基于真实用户画像与推理结果演示</p>
      </div>
      <div className="flex items-center gap-2">
        {(['GM', 'AO'] as const).map((k) => (
          <button
            key={k}
            onClick={() => onSwitchDataset(k)}
            className={`rounded-lg px-3 py-1.5 text-sm font-medium transition ${
              datasetKey === k
                ? 'bg-blue-600 text-white shadow-sm'
                : 'bg-white text-gray-600 ring-1 ring-gray-200 hover:bg-gray-50'
            }`}
          >
            {k === 'GM' ? 'GM 数据集' : 'AO 数据集'}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
        <div className="lg:col-span-2">
          <HistoryPanel
            users={dataset.users}
            selectedId={currentUser.id}
            onSelect={handleSelectUser}
            onRandom={handleRandomUser}
          />
          <button
            onClick={handleGenerate}
            disabled={loading}
            className="mt-4 w-full rounded-xl bg-blue-600 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading ? '生成中...' : '生成推荐 →'}
          </button>
        </div>

        <div className="lg:col-span-3">
          <ResultPanel
            result={result}
            loading={loading}
            feedback={feedback}
            onFeedback={setFeedback}
          />
        </div>
      </div>
    </div>
  )
}
