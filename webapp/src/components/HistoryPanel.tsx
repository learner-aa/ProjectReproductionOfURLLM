import type { DemoUser } from '../types'
import { KindBadge } from './KindBadge'
import { History } from 'lucide-react'

interface Props {
  users: DemoUser[]
  selectedId: string
  onSelect: (id: string) => void
  onRandom: () => void
}

export function HistoryPanel({ users, selectedId, onSelect, onRandom }: Props) {
  const user = users.find((u) => u.id === selectedId) ?? users[0]

  return (
    <div className="flex h-full flex-col rounded-2xl border border-gray-200 bg-white shadow-sm">
      <div className="flex items-center justify-between border-b border-gray-100 px-5 py-4">
        <div className="flex items-center gap-2 text-sm font-semibold text-gray-800">
          <History size={16} className="text-gray-400" />
          用户历史交互
          <span className="rounded-full bg-blue-50 px-2 py-0.5 text-xs font-medium text-blue-600">
            {users.length} 位可选
          </span>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={selectedId}
            onChange={(e) => onSelect(e.target.value)}
            className="rounded-lg border border-gray-200 bg-gray-50 px-2 py-1 text-sm text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-300"
          >
            {users.map((u) => (
              <option key={u.id} value={u.id}>
                {u.id}
              </option>
            ))}
          </select>
          <button
            onClick={onRandom}
            className="rounded-lg border border-gray-200 px-2 py-1 text-sm text-gray-600 hover:bg-gray-50"
            title="随机切换用户"
          >
            🔀
          </button>
        </div>
      </div>

      <div className="flex-1 space-y-2 overflow-y-auto px-5 py-4">
        {user?.history.map((item, idx) => (
          <div
            key={idx}
            className="flex items-center justify-between rounded-xl border border-gray-100 bg-gray-50/60 px-3 py-2.5"
          >
            <div className="flex items-center gap-2 text-sm text-gray-700">
              <span className="text-xs text-gray-400 tabular-nums">{idx + 1}</span>
              <span className="line-clamp-1">{item.title}</span>
            </div>
            <KindBadge kind={item.kind} />
          </div>
        ))}
      </div>

      <div className="border-t border-gray-100 px-5 py-3 text-xs text-gray-400">
        共 {user?.history.length ?? 0} 条交互记录 · 按时间顺序排列
      </div>
    </div>
  )
}
