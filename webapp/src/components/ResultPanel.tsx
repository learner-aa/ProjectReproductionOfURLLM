import type { RecommendationResult } from '../types'
import { KindBadge } from './KindBadge'
import { Sparkles, ThumbsUp, ThumbsDown, Users, Loader2 } from 'lucide-react'

interface Props {
  result: RecommendationResult | null
  loading: boolean
  feedback: 'up' | 'down' | null
  onFeedback: (v: 'up' | 'down') => void
}

export function ResultPanel({ result, loading, feedback, onFeedback }: Props) {
  return (
    <div className="flex flex-col gap-4">
      <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
        <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-gray-800">
          <Sparkles size={16} className="text-blue-500" />
          推荐结果
        </div>

        {loading && (
          <div className="flex items-center gap-2 py-8 text-sm text-gray-400">
            <Loader2 size={16} className="animate-spin" />
            正在生成推荐...
          </div>
        )}

        {!loading && !result && (
          <div className="py-8 text-center text-sm text-gray-400">
            点击下方「生成推荐」查看结果
          </div>
        )}

        {!loading && result && (
          <>
            <div className="flex items-center gap-2">
              <span className="text-lg font-semibold text-gray-900">{result.title}</span>
              <KindBadge kind={result.kind} />
            </div>
            <p className="mt-3 rounded-xl bg-blue-50/60 p-3 text-sm leading-relaxed text-gray-700">
              {result.reason}
            </p>
            <div className="mt-4 flex items-center gap-2">
              <button
                onClick={() => onFeedback('up')}
                className={`flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-sm transition ${
                  feedback === 'up'
                    ? 'border-emerald-300 bg-emerald-50 text-emerald-700'
                    : 'border-gray-200 text-gray-600 hover:bg-gray-50'
                }`}
              >
                <ThumbsUp size={14} /> 采纳
              </button>
              <button
                onClick={() => onFeedback('down')}
                className={`flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-sm transition ${
                  feedback === 'down'
                    ? 'border-rose-300 bg-rose-50 text-rose-700'
                    : 'border-gray-200 text-gray-600 hover:bg-gray-50'
                }`}
              >
                <ThumbsDown size={14} /> 反馈
              </button>
            </div>
          </>
        )}
      </div>

      {!loading && result && (
        <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
          <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-gray-800">
            <Users size={16} className="text-gray-400" />
            相似用户参考 (Top-{result.similarUsers.length})
          </div>
          <div className="space-y-3">
            {result.similarUsers.map((su) => (
              <div key={su.id} className="rounded-xl border border-gray-100 bg-gray-50/60 p-3">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-medium text-gray-700">{su.id}</span>
                  <span className="text-xs text-gray-400">相似度 {su.similarity.toFixed(2)}</span>
                </div>
                <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-gray-200">
                  <div
                    className="h-full rounded-full bg-blue-400"
                    style={{ width: `${su.similarity * 100}%` }}
                  />
                </div>
                <div className="mt-2 line-clamp-1 text-xs text-gray-500">
                  最近交互：{su.recentTitles.join(' / ')}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
