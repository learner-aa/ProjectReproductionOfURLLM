import { useState } from 'react'
import type { Dataset, DemoUser, RecommendationResult } from '../types'
import { HistoryPanel } from '../components/HistoryPanel'
import { ResultPanel } from '../components/ResultPanel'

interface Props {
  dataset: Dataset
  currentUser: DemoUser
  onSelectUser: (id: string) => void
  onRandomUser: () => void
}

export function Workbench({ dataset, currentUser, onSelectUser, onRandomUser }: Props) {
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
    // 接入 enhancement/src/llm_inference.py 对应的推理结果(已预计算)
    window.setTimeout(() => {
      setResult(currentUser.result)
      setLoading(false)
    }, 900)
  }

  return (
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
  )
}
