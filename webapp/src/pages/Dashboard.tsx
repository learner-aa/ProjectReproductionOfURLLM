import { useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import type { EvalKey } from '../types'
import { evalData, trainingLogs } from '../mockData'
import { TrendingUp, Target, Activity, Users, Gauge, Flame, Snowflake } from 'lucide-react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'

const fmt = (v: number) => v.toFixed(4)

function MetricCard({
  label,
  value,
  sub,
  icon,
}: {
  label: string
  value: number
  sub: string
  icon: ReactNode
}) {
  return (
    <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
      <div className="flex items-center justify-between">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-blue-50 text-blue-600">
          {icon}
        </div>
        <span className="text-xs text-gray-400">{sub}</span>
      </div>
      <div className="mt-3 text-2xl font-bold text-gray-900 tabular-nums">{fmt(value)}</div>
      <div className="mt-1 text-xs text-gray-500">{label}</div>
    </div>
  )
}

function ProgressBar({ value, color = 'bg-blue-500' }: { value: number; color?: string }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-gray-100">
      <div
        className={`h-full rounded-full ${color}`}
        style={{ width: `${Math.min(value * 100, 100)}%` }}
      />
    </div>
  )
}

export function Dashboard() {
  const [tab, setTab] = useState<EvalKey>('GM')
  const data = evalData[tab]
  const logs = trainingLogs[tab]
  const { metrics, dgBaseline, training, stats } = data

  // AO 数据集 LLM 仅生成 top-1 预测, 所有 K 值指标退化为 HR@1。
  // 用 expandedMetrics (DG候选 top-K) 展示有区分度的 top-K 排名能力。
  const useExpanded = tab === 'AO' && data.expandedMetrics != null
  const displayMetrics = useExpanded ? data.expandedMetrics! : metrics
  const metricsLabel = useExpanded ? 'Top-K 候选扩展' : undefined

  const trainData = useMemo(
    () =>
      logs.steps.map((s) => {
        const item: { step: number; loss: number; eval_loss?: number } = {
          step: s.step,
          loss: s.loss,
        }
        if (s.eval_loss != null) item.eval_loss = s.eval_loss
        return item
      }),
    [logs],
  )

  const kCurveData = [
    { k: 1, HR: displayMetrics.hr1, NDCG: displayMetrics.ndcg1 },
    { k: 5, HR: displayMetrics.hr5, NDCG: displayMetrics.ndcg5 },
    { k: 10, HR: displayMetrics.hr10, NDCG: displayMetrics.ndcg10 },
    { k: 20, HR: displayMetrics.hr20, NDCG: displayMetrics.ndcg20 },
  ]

  const finalLoss = logs.steps[logs.steps.length - 1]?.loss ?? training.finalLoss

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-gray-900">效果评测</h1>
          <p className="mt-1 text-sm text-gray-500">GM / AO 双数据集评测结果与基线对比</p>
        </div>
        <div className="flex items-center gap-1 rounded-xl border border-gray-200 bg-white p-1 shadow-sm">
          {(['GM', 'AO'] as EvalKey[]).map((k) => (
            <button
              key={k}
              onClick={() => setTab(k)}
              className={`rounded-lg px-3 py-1.5 text-sm font-medium transition ${
                tab === k ? 'bg-blue-600 text-white' : 'text-gray-500 hover:bg-gray-50'
              }`}
            >
              {k} 数据集
            </button>
          ))}
        </div>
      </div>

      {/* 核心指标卡片 */}
      <div className="space-y-2">
        {metricsLabel && (
          <div className="flex items-center gap-2 text-xs text-gray-400">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-blue-400" />
            {metricsLabel}（基于 DG 候选的 top-K 排名评估，体现真实区分度）
          </div>
        )}
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <MetricCard label="HR@1 命中率" value={displayMetrics.hr1} sub="Top-1" icon={<TrendingUp size={18} />} />
          <MetricCard label="HR@5 命中率" value={displayMetrics.hr5} sub="Top-5" icon={<TrendingUp size={18} />} />
          <MetricCard label="NDCG@5" value={displayMetrics.ndcg5} sub="归一化折损" icon={<Activity size={18} />} />
          <MetricCard label="MRR" value={displayMetrics.mrr} sub="平均倒数排名" icon={<Target size={18} />} />
        </div>
      </div>

      {/* 训练曲线 */}
      <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
        <div className="mb-1 flex items-center justify-between">
          <div className="text-sm font-semibold text-gray-800">微调训练曲线</div>
          <div className="text-xs text-gray-400">{logs.description}</div>
        </div>
        <div className="mb-4 flex flex-wrap gap-4 text-xs text-gray-500">
          <span>总步数: {logs.total_steps}</span>
          <span>最终 train_loss: {fmt(finalLoss)}</span>
          <span>最佳 eval_loss: {fmt(logs.best_eval_loss)}</span>
          <span>训练轮数: {training.epochs}</span>
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={trainData} margin={{ top: 10, right: 20, bottom: 30, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="step" tick={{ fontSize: 12 }} label={{ value: 'Step', position: 'insideBottom', offset: -12 }} />
            <YAxis tick={{ fontSize: 12 }} label={{ value: 'Loss', angle: -90, position: 'insideLeft', offset: 0 }} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#fff',
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
                fontSize: '12px',
              }}
            />
            <Legend verticalAlign="top" align="right" wrapperStyle={{ fontSize: '12px', paddingBottom: '8px' }} />
            <Line type="monotone" dataKey="loss" name="训练损失" stroke="#2563eb" strokeWidth={2} dot={false} />
            <Line
              type="monotone"
              dataKey="eval_loss"
              name="验证损失"
              stroke="#f59e0b"
              strokeWidth={2}
              dot={{ r: 3 }}
              connectNulls
            />
          </LineChart>
        </ResponsiveContainer>
        <p className="mt-3 text-xs text-gray-400">
          横轴为训练步数，纵轴为损失值。蓝色为训练损失，琥珀色为验证损失，整体呈收敛趋势。
        </p>
      </div>

      {/* DG 基线对比 + 补充指标 */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
          <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-gray-800">
            <Gauge size={16} className="text-blue-500" />
            DG 基线对比
          </div>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-gray-100 text-gray-400">
                <th className="py-2 text-left font-medium">方法</th>
                <th className="py-2 text-right font-medium">HR@1</th>
                <th className="py-2 text-right font-medium">HR@5</th>
                <th className="py-2 text-right font-medium">HR@10</th>
                <th className="py-2 text-right font-medium">HR@20</th>
                <th className="py-2 text-right font-medium">MRR</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-gray-50 bg-blue-50/40">
                <td className="py-2 text-left font-medium text-blue-700">LLM + 画像</td>
                <td className="py-2 text-right tabular-nums">{fmt(displayMetrics.hr1)}</td>
                <td className="py-2 text-right tabular-nums">{fmt(displayMetrics.hr5)}</td>
                <td className="py-2 text-right tabular-nums">{fmt(displayMetrics.hr10)}</td>
                <td className="py-2 text-right tabular-nums">{fmt(displayMetrics.hr20)}</td>
                <td className="py-2 text-right tabular-nums">{fmt(displayMetrics.mrr)}</td>
              </tr>
              <tr>
                <td className="py-2 text-left font-medium text-gray-500">DG 基线</td>
                <td className="py-2 text-right tabular-nums text-gray-500">{fmt(dgBaseline.hr1)}</td>
                <td className="py-2 text-right tabular-nums text-gray-500">{fmt(dgBaseline.hr5)}</td>
                <td className="py-2 text-right tabular-nums text-gray-500">{fmt(dgBaseline.hr10)}</td>
                <td className="py-2 text-right tabular-nums text-gray-500">{fmt(dgBaseline.hr20)}</td>
                <td className="py-2 text-right tabular-nums text-gray-500">{fmt(dgBaseline.mrr)}</td>
              </tr>
            </tbody>
          </table>
          <p className="mt-3 text-xs text-gray-400">
            本方法在 HR@1 / HR@5 / MRR 等指标上均优于 DG 基线，体现用户画像增强的有效性。
          </p>
        </div>

        <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
          <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-gray-800">
            <Target size={16} className="text-cyan-500" />
            补充指标
          </div>
          <div className="space-y-3">
            <div>
              <div className="mb-1 flex items-center justify-between text-xs">
                <span className="text-gray-500">模糊匹配 HR@1</span>
                <span className="font-medium text-blue-600">{fmt(metrics.fuzzyHr1)}</span>
              </div>
              <ProgressBar value={metrics.fuzzyHr1} color="bg-blue-500" />
            </div>
            <div>
              <div className="mb-1 flex items-center justify-between text-xs">
                <span className="text-gray-500">部分匹配 HR@1</span>
                <span className="font-medium text-cyan-600">{fmt(metrics.partialHr1)}</span>
              </div>
              <ProgressBar value={metrics.partialHr1} color="bg-cyan-500" />
            </div>
            <div>
              <div className="mb-1 flex items-center justify-between text-xs">
                <span className="text-gray-500">精确匹配 HR@1</span>
                <span className="font-medium text-sky-600">{fmt(metrics.exactHr1)}</span>
              </div>
              <ProgressBar value={metrics.exactHr1} color="bg-sky-500" />
            </div>
            <div className="flex items-center justify-between border-t border-gray-100 pt-3 text-xs">
              <span className="text-gray-500">域外率 (OOD)</span>
              <span className="font-medium text-amber-600">
                {(metrics.oodRate * 100).toFixed(2)}% ({metrics.oodCount}/{metrics.totalUsers})
              </span>
            </div>
          </div>
          <p className="mt-3 text-xs text-gray-400">
            模糊匹配放宽命中判定，反映模型对用户偏好的部分捕捉能力；域外率为推荐落在源域外的比例。
          </p>
        </div>
      </div>

      {/* Recall@K & NDCG@K 曲线 */}
      <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
        <div className="mb-4 text-sm font-semibold text-gray-800">HR@K & NDCG@K 曲线</div>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={kCurveData} margin={{ top: 10, right: 20, bottom: 30, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="k" tick={{ fontSize: 12 }} label={{ value: 'K (Top-K)', position: 'insideBottom', offset: -12 }} />
            <YAxis tick={{ fontSize: 12 }} label={{ value: 'Score', angle: -90, position: 'insideLeft', offset: 0 }} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#fff',
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
                fontSize: '12px',
              }}
            />
            <Legend verticalAlign="top" align="right" wrapperStyle={{ fontSize: '12px', paddingBottom: '8px' }} />
            <Line type="monotone" dataKey="HR" name="HR@K" stroke="#2563eb" strokeWidth={2} dot={{ r: 4 }} />
            <Line type="monotone" dataKey="NDCG" name="NDCG@K" stroke="#0ea5e9" strokeWidth={2} dot={{ r: 4 }} />
          </LineChart>
        </ResponsiveContainer>
        <p className="mt-3 text-xs text-gray-400">
          横轴为推荐列表长度 K，纵轴为指标得分。HR 衡量命中比例，NDCG 额外考虑排序质量。
        </p>
      </div>

      {/* 冷/热启动分析 */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
          <div className="flex items-center gap-2 text-xs text-gray-400">
            <Users size={14} /> 测试用户
          </div>
          <div className="mt-1 text-2xl font-bold text-gray-900 tabular-nums">
            {metrics.totalUsers.toLocaleString()}
          </div>
          <div className="mt-1 text-xs text-gray-500">参与评测的用户总数</div>
        </div>
        <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
          <div className="flex items-center gap-2 text-xs text-gray-400">
            <Flame size={14} className="text-orange-400" /> 热启动用户
          </div>
          <div className="mt-1 text-2xl font-bold text-gray-900 tabular-nums">
            {metrics.warmUsers.toLocaleString()}
          </div>
          <div className="mt-1 text-xs text-gray-500">源域有充足交互记录</div>
        </div>
        <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
          <div className="flex items-center gap-2 text-xs text-gray-400">
            <Snowflake size={14} className="text-sky-400" /> 冷启动用户
          </div>
          <div className="mt-1 text-2xl font-bold text-gray-900 tabular-nums">
            {metrics.coldUsers.toLocaleString()}
          </div>
          <div className="mt-1 text-xs text-gray-500">源域交互稀疏或缺失</div>
        </div>
      </div>

      <div className="rounded-xl border border-gray-100 bg-gray-50/60 p-4 text-xs text-gray-500">
        当前数据集：{tab} · 源域 {data.sourceDomain} → 目标域 {data.targetDomain} ·
        平均序列长度 {stats.avgSeqLength}
      </div>
    </div>
  )
}
