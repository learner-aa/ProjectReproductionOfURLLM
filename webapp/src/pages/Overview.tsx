import type { ReactNode } from 'react'
import { evalData } from '../mockData'
import type { EvalDataset, EvalKey } from '../types'
import { TrendingUp, Target, Users, ArrowRight, Database, Package } from 'lucide-react'

const fmt = (v: number) => v.toFixed(4)

function CoreResultCard({
  label,
  value,
  baseline,
  baselineLabel,
  isCount,
  highlight,
  icon,
}: {
  label: string
  value: number
  baseline?: number
  baselineLabel?: string
  isCount?: boolean
  highlight?: boolean
  icon: ReactNode
}) {
  const display = isCount ? value.toLocaleString() : fmt(value)
  return (
    <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
      <div className="flex items-center justify-between">
        <div
          className={`flex h-9 w-9 items-center justify-center rounded-lg ${
            highlight ? 'bg-blue-50 text-blue-600' : 'bg-cyan-50 text-cyan-600'
          }`}
        >
          {icon}
        </div>
        <span className="text-xs text-gray-400">{label}</span>
      </div>
      <div className="mt-3 text-2xl font-bold text-gray-900 tabular-nums">{display}</div>
      <div className="mt-2 flex items-center gap-1 text-xs text-gray-500">
        {baseline !== undefined && baselineLabel !== undefined && (
          <>
            <span className="rounded bg-gray-100 px-1.5 py-0.5 text-gray-500">
              {baselineLabel} {fmt(baseline)}
            </span>
            {value > baseline && (
              <span className="font-medium text-blue-600">显著优于基线</span>
            )}
          </>
        )}
        {isCount && (
          <span className="rounded bg-gray-100 px-1.5 py-0.5 text-gray-500">{baselineLabel}</span>
        )}
      </div>
    </div>
  )
}

function DatasetOverviewCard({ data, name }: { data: EvalDataset; name: string }) {
  const { stats, sourceDomain, targetDomain } = data
  const interactionText = stats.sourceInteractions && stats.targetInteractions
    ? `源 ${stats.sourceInteractions.toLocaleString()} / 目标 ${stats.targetInteractions.toLocaleString()}`
    : stats.totalInteractions
      ? stats.totalInteractions.toLocaleString()
      : '—'
  return (
    <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
      <div className="flex items-center justify-between border-b border-gray-100 pb-3">
        <div className="text-sm font-semibold text-gray-800">{name} 数据集</div>
        <span className="text-xs text-gray-400">{data.label}</span>
      </div>

      <div className="mt-3 flex items-center gap-2 text-xs text-gray-500">
        <span className="rounded bg-blue-50 px-2 py-0.5 font-medium text-blue-700">{sourceDomain}</span>
        <ArrowRight size={12} className="text-gray-400" />
        <span className="rounded bg-cyan-50 px-2 py-0.5 font-medium text-cyan-700">{targetDomain}</span>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-x-4 gap-y-3 text-sm">
        <div>
          <div className="flex items-center gap-1 text-xs text-gray-400">
            <Users size={12} /> 用户总数
          </div>
          <div className="mt-0.5 font-semibold text-gray-900 tabular-nums">
            {stats.numUsers.toLocaleString()}
          </div>
        </div>
        <div>
          <div className="flex items-center gap-1 text-xs text-gray-400">
            <Users size={12} /> 测试用户
          </div>
          <div className="mt-0.5 font-semibold text-gray-900 tabular-nums">
            {stats.testUsers.toLocaleString()}
          </div>
        </div>
        <div>
          <div className="flex items-center gap-1 text-xs text-gray-400">
            <Package size={12} /> 源域物品
          </div>
          <div className="mt-0.5 font-semibold text-gray-900 tabular-nums">
            {stats.sourceItems.toLocaleString()}
          </div>
        </div>
        <div>
          <div className="flex items-center gap-1 text-xs text-gray-400">
            <Package size={12} /> 目标域物品
          </div>
          <div className="mt-0.5 font-semibold text-gray-900 tabular-nums">
            {stats.targetItems.toLocaleString()}
          </div>
        </div>
        <div className="col-span-2">
          <div className="flex items-center gap-1 text-xs text-gray-400">
            <Database size={12} /> 交互数
          </div>
          <div className="mt-0.5 font-semibold text-gray-900 tabular-nums">{interactionText}</div>
        </div>
        <div className="col-span-2">
          <div className="flex items-center gap-1 text-xs text-gray-400">训练指令数</div>
          <div className="mt-0.5 font-semibold text-gray-900 tabular-nums">
            {stats.trainInstructions.toLocaleString()}
          </div>
        </div>
      </div>
    </div>
  )
}

export function Overview() {
  const gm = evalData.GM
  const ao = evalData.AO
  const totalTestUsers = gm.metrics.totalUsers + ao.metrics.totalUsers

  return (
    <div className="space-y-6">
      {/* 项目标题 */}
      <div className="rounded-2xl border border-gray-200 bg-gradient-to-br from-blue-50 to-white p-6 shadow-sm">
        <h1 className="text-2xl font-bold text-gray-900">URLLM 跨域序列推荐</h1>
        <p className="mt-2 text-sm leading-relaxed text-gray-600">
          基于大语言模型与用户画像增强的跨域推荐方法
        </p>
        <p className="mt-3 text-xs leading-relaxed text-gray-500">
          在 GM 与 AO 两个跨域数据集上完成从源域行为到目标域推荐的迁移评测，
          覆盖 {totalTestUsers.toLocaleString()} 位测试用户。
        </p>
      </div>

      {/* 核心成果卡片 */}
      <div>
        <div className="mb-3 text-sm font-semibold text-gray-800">核心成果</div>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <CoreResultCard
            label="GM HR@1"
            value={gm.metrics.hr1}
            baseline={gm.dgBaseline.hr1}
            baselineLabel="DG 基线"
            highlight
            icon={<TrendingUp size={18} />}
          />
          <CoreResultCard
            label="AO HR@1"
            value={ao.metrics.hr1}
            baseline={ao.dgBaseline.hr1}
            baselineLabel="DG 基线"
            highlight
            icon={<TrendingUp size={18} />}
          />
          <CoreResultCard
            label="GM MRR"
            value={gm.metrics.mrr}
            baseline={gm.dgBaseline.mrr}
            baselineLabel="DG 基线"
            icon={<Target size={18} />}
          />
          <CoreResultCard
            label="测试用户"
            value={totalTestUsers}
            isCount
            baselineLabel={`GM ${gm.metrics.totalUsers} + AO ${ao.metrics.totalUsers}`}
            icon={<Users size={18} />}
          />
        </div>
      </div>

      {/* 双数据集概览 */}
      <div>
        <div className="mb-3 text-sm font-semibold text-gray-800">双数据集概览</div>
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {(['GM', 'AO'] as EvalKey[]).map((k) => (
            <DatasetOverviewCard key={k} name={k} data={evalData[k]} />
          ))}
        </div>
      </div>

      {/* 方法一句话 */}
      <div className="rounded-2xl border border-blue-100 bg-blue-50/50 p-5 text-sm leading-relaxed text-gray-700">
        <span className="font-semibold text-blue-700">方法概述：</span>
        通过用户画像构建与大语言模型微调，将源域行为模式迁移至目标域，实现跨域个性化推荐。
      </div>
    </div>
  )
}
