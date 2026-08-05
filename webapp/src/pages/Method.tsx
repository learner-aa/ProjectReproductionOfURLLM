import { datasets, evalData } from '../mockData'
import {
  Database,
  Package,
  Users,
  GitBranch,
  Cpu,
  TrendingUp,
  FileText,
  ArrowRight,
  ArrowDown,
  Sparkles,
} from 'lucide-react'
import type { ReactNode } from 'react'

interface Stage {
  name: string
  desc: string
  icon: typeof Database
  core?: boolean
}

const PIPELINE_STAGES: Stage[] = [
  { name: '数据预处理', desc: '解析跨域交互日志，划分训练 / 验证 / 测试集', icon: Database },
  { name: '物品属性提取', desc: '大语言模型提取物品语义属性，构建属性索引', icon: Package },
  { name: '用户画像构建', desc: '聚合行为特征与语义偏好，生成结构化画像', icon: Users, core: true },
  { name: '指令数据构建', desc: '将画像与目标物品组织为指令微调样本', icon: GitBranch },
  { name: '模型微调', desc: '参数高效微调学习用户偏好模式', icon: Cpu, core: true },
  { name: '推理生成', desc: '基于画像生成目标域候选推荐', icon: TrendingUp },
  { name: '评估评测', desc: 'HR@K / NDCG@K / MRR 与基线对比', icon: FileText },
]

function ProfileField({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-3 border-b border-gray-100 py-2 last:border-0">
      <span className="text-xs text-gray-400">{label}</span>
      <span className="text-right text-sm font-medium text-gray-700">{value}</span>
    </div>
  )
}

export function Method() {
  const gm = evalData.GM
  const demoUser = datasets[0].users[0]
  const history = demoUser.history
  const profile = (demoUser as { profile?: { totalInteractions: number; domainXCount: number; domainYCount: number; preferredAttributes: { name: string; weight: number }[] } }).profile
  const totalInteractions = profile?.totalInteractions ?? history.length
  const domainDist = profile
    ? `${gm.sourceDomain} ${profile.domainXCount} / ${gm.targetDomain} ${profile.domainYCount}`
    : Object.entries(
        history.reduce<Record<string, number>>((acc, h) => {
          acc[h.kind] = (acc[h.kind] ?? 0) + 1
          return acc
        }, {}),
      )
        .map(([k, c]) => `${k} ${c}`)
        .join(' / ')
  const topAttrs = profile?.preferredAttributes?.slice(0, 4) ?? []

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-bold text-gray-900">技术方法与流程</h1>
        <p className="mt-1 text-sm text-gray-500">
          从原始交互到跨域推荐的完整 Pipeline，核心创新在于用户画像构建与模型微调两个阶段。
        </p>
      </div>

      {/* Pipeline 7 阶段 */}
      <div className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm">
        <div className="mb-4 text-sm font-semibold text-gray-800">Pipeline 流程（7 阶段）</div>
        <div className="flex flex-col gap-2 lg:flex-row lg:items-stretch">
          {PIPELINE_STAGES.map((stage, idx) => {
            const Icon = stage.icon
            return (
              <div key={stage.name} className="flex flex-1 flex-col lg:flex-row lg:items-center">
                <div
                  className={`flex-1 rounded-xl border p-3 transition ${
                    stage.core
                      ? 'border-blue-200 bg-blue-50/60 ring-1 ring-blue-100'
                      : 'border-gray-100 bg-gray-50/60'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div
                      className={`flex h-7 w-7 items-center justify-center rounded-lg ${
                        stage.core ? 'bg-blue-100 text-blue-600' : 'bg-gray-100 text-gray-500'
                      }`}
                    >
                      <Icon size={14} />
                    </div>
                    <span className="text-[10px] text-gray-400">{idx + 1}</span>
                  </div>
                  <div className="mt-2 flex items-center gap-1 text-sm font-semibold text-gray-800">
                    {stage.name}
                    {stage.core && (
                      <span className="rounded bg-blue-600 px-1 py-0.5 text-[9px] font-medium text-white">
                        核心创新
                      </span>
                    )}
                  </div>
                  <div className="mt-1 text-[11px] leading-relaxed text-gray-500">{stage.desc}</div>
                </div>
                {idx < PIPELINE_STAGES.length - 1 && (
                  <div className="flex items-center justify-center px-1 py-1 text-gray-300">
                    <ArrowRight size={14} className="hidden lg:block" />
                    <ArrowDown size={14} className="lg:hidden" />
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* 用户画像样例 */}
        <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
          <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-gray-800">
            <Users size={16} className="text-blue-500" />
            用户画像样例
          </div>
          <div className="rounded-xl border border-gray-100 bg-gray-50/60 p-4">
            <div className="mb-2 flex items-center justify-between">
              <span className="text-xs text-gray-400">用户 ID</span>
              <span className="text-sm font-semibold text-gray-800">{demoUser.id}</span>
            </div>
            <ProfileField label="交互次数" value={totalInteractions} />
            <ProfileField label="域分布" value={domainDist} />
            <ProfileField label="序列长度" value={`${totalInteractions} 条`} />
            <ProfileField
              label="偏好属性"
              value={
                <span className="flex flex-wrap justify-end gap-1">
                  {topAttrs.map((attr, idx) => (
                    <span
                      key={attr.name}
                      className={`rounded px-1.5 py-0.5 text-[11px] ${
                        idx % 2 === 0
                          ? 'bg-blue-50 text-blue-700'
                          : 'bg-cyan-50 text-cyan-700'
                      }`}
                    >
                      {attr.name}
                    </span>
                  ))}
                </span>
              }
            />
            <ProfileField
              label="活跃域"
              value={<span className="text-blue-600">{gm.sourceDomain}</span>}
            />
          </div>
          <p className="mt-3 text-xs text-gray-400">
            画像由源域行为聚合得到，作为微调与推理阶段的输入上下文，刻画用户兴趣偏好。
          </p>
        </div>

        {/* 跨域推荐示意 */}
        <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
          <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-gray-800">
            <Sparkles size={16} className="text-cyan-500" />
            跨域推荐示意
          </div>
          <div className="flex flex-col gap-3">
            <div className="rounded-xl border border-blue-100 bg-blue-50/40 p-3">
              <div className="text-[11px] font-medium text-blue-600">源域 · {gm.sourceDomain}</div>
              <div className="mt-1 line-clamp-1 text-xs text-gray-600">
                {history.slice(0, 3).map((h) => h.title).join(' / ')}
              </div>
              <div className="mt-1 text-[10px] text-gray-400">用户历史交互</div>
            </div>

            <div className="flex items-center justify-center text-gray-300">
              <ArrowDown size={16} />
            </div>

            <div className="rounded-xl border border-gray-200 bg-gray-50/60 p-3 text-center">
              <div className="text-[11px] font-medium text-gray-600">用户画像</div>
              <div className="mt-1 text-[10px] text-gray-400">
                行为特征 + 语义偏好 → 结构化画像
              </div>
            </div>

            <div className="flex items-center justify-center text-gray-300">
              <ArrowDown size={16} />
            </div>

            <div className="rounded-xl border border-cyan-100 bg-cyan-50/40 p-3">
              <div className="text-[11px] font-medium text-cyan-600">目标域 · {gm.targetDomain}</div>
              <div className="mt-1 flex items-center justify-between">
                <span className="line-clamp-1 text-xs text-gray-700">{demoUser.result.title}</span>
                <span className="ml-2 shrink-0 rounded bg-white px-1.5 py-0.5 text-[10px] text-cyan-700">
                  推荐结果
                </span>
              </div>
            </div>
          </div>
          <p className="mt-3 text-xs text-gray-400">
            模型依据源域画像在目标域生成候选推荐，完成跨域兴趣迁移。
          </p>
        </div>
      </div>
    </div>
  )
}
