import type { RetrievalCandidate } from '../types'
import { CheckCircle2 } from 'lucide-react'

// 检索候选池排行：按相似度从高到低排列，虚线以上是被选中拼进 LLM prompt 的 Top-K
export function RetrievalRanking({ pool }: { pool: RetrievalCandidate[] }) {
  const sorted = [...pool].sort((a, b) => b.similarity - a.similarity)
  const selectedCount = sorted.filter((c) => c.selected).length

  return (
    <div className="space-y-2">
      {sorted.map((c, idx) => (
        <div key={c.id}>
          {idx === selectedCount && (
            <div className="my-2 flex items-center gap-2 text-xs text-gray-400">
              <div className="h-px flex-1 border-t border-dashed border-gray-300" />
              以下候选相似度不足，未被选入 Top-{selectedCount}
              <div className="h-px flex-1 border-t border-dashed border-gray-300" />
            </div>
          )}
          <div
            className={`rounded-xl border p-3 transition ${
              c.selected
                ? 'border-blue-200 bg-blue-50/50'
                : 'border-gray-100 bg-gray-50/40 opacity-70'
            }`}
          >
            <div className="flex items-center justify-between text-sm">
              <span className="flex items-center gap-1.5 font-medium text-gray-700">
                <span className="text-xs text-gray-400 tabular-nums">#{idx + 1}</span>
                {c.id}
                {c.selected && <CheckCircle2 size={14} className="text-blue-500" />}
              </span>
              <span className="text-xs text-gray-400">相似度 {c.similarity.toFixed(3)}</span>
            </div>
            <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-gray-200">
              <div
                className={`h-full rounded-full ${c.selected ? 'bg-blue-500' : 'bg-gray-300'}`}
                style={{ width: `${c.similarity * 100}%` }}
              />
            </div>
            <div className="mt-2 line-clamp-1 text-xs text-gray-500">
              最近交互：{c.recentTitles.join(' / ')}
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
