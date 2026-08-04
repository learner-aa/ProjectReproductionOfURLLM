import type { Dataset, DemoUser } from '../types'
import { GraphVisual } from '../components/GraphVisual'
import { RetrievalRanking } from '../components/RetrievalRanking'
import { HistoryPanel } from '../components/HistoryPanel'
import { Waypoints, ArrowRight } from 'lucide-react'

interface Props {
  dataset: Dataset
  currentUser: DemoUser
  onSelectUser: (id: string) => void
  onRandomUser: () => void
}

export function RetrievalView({ dataset, currentUser, onSelectUser, onRandomUser }: Props) {
  const selectedCount = currentUser.retrievalPool.filter((c) => c.selected).length

  return (
    <div className="space-y-6">
      {/* 用户选择 + 交互图 */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
        <div className="lg:col-span-2">
          <HistoryPanel
            users={dataset.users}
            selectedId={currentUser.id}
            onSelect={onSelectUser}
            onRandom={onRandomUser}
          />
        </div>

        <div className="lg:col-span-3 flex flex-col gap-4">
          <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
            <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-gray-800">
              <Waypoints size={16} className="text-blue-500" />
              当前用户交互图
            </div>
            <GraphVisual history={currentUser.history} />
            <p className="mt-2 text-xs text-gray-400">
              节点 = 交互过的物品（按颜色区分 Movie / Game），边 = 序列上的先后关系。
              系统从这条交互序列中提取用户的行为特征与语义偏好画像。
            </p>
          </div>

          <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
            <div className="mb-2 text-sm font-semibold text-gray-800">检索流程说明</div>
            <div className="flex flex-wrap items-center gap-2 text-xs text-gray-500">
              <span className="rounded-lg bg-gray-100 px-2 py-1">用户交互历史</span>
              <ArrowRight size={12} />
              <span className="rounded-lg bg-gray-100 px-2 py-1">物品 Jaccard 相似度</span>
              <ArrowRight size={12} />
              <span className="rounded-lg bg-gray-100 px-2 py-1">按相似度排序</span>
              <ArrowRight size={12} />
              <span className="rounded-lg bg-blue-100 px-2 py-1 text-blue-700">
                取 Top-{selectedCount} 相似用户拼入 LLM Prompt
              </span>
            </div>
            <p className="mt-3 text-xs leading-relaxed text-gray-400">
              系统从用户交互序列中提取行为特征与语义偏好画像，再结合相似用户信息拼成指令输入大模型。
            </p>
          </div>
        </div>
      </div>

      {/* 候选池排行 */}
      <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
        <div className="mb-1 flex items-center justify-between">
          <div className="text-sm font-semibold text-gray-800">
            相似度检索候选池（共 {currentUser.retrievalPool.length} 个候选）
          </div>
          <div className="text-xs text-gray-400">按相似度降序排列</div>
        </div>
        <p className="mb-4 text-xs text-gray-400">
          虚线以上为被选入 Top-{selectedCount} 并拼接进最终 Prompt 的相似用户，虚线以下为候选池中相似度不足、未被采用的用户。
        </p>
        <RetrievalRanking pool={currentUser.retrievalPool} />
      </div>
    </div>
  )
}
