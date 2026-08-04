import type { DatasetKey } from '../types'
import { metricsSnapshots, extraMetrics, trainingLogs } from '../mockData'
import { TrendingUp, Users, BarChart3, Activity, Target, Globe } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

interface Props {
  datasetKey: DatasetKey
}

export function Dashboard({ datasetKey }: Props) {
  const metrics = metricsSnapshots[datasetKey]
  const logs = trainingLogs as any

  // 训练曲线数据
  const trainData = logs.steps.map((s: any) => ({
    step: s.step,
    loss: s.loss,
    eval_loss: s.eval_loss ?? null,
  }))

  // K 值曲线数据
  const chartData = Object.keys(metrics.recallAtK).map((k) => ({
    k: Number(k),
    Recall: metrics.recallAtK[Number(k)],
    NDCG: metrics.ndcgAtK[Number(k)],
  }))

  return (
    <div className="space-y-6">
      {/* 核心指标卡片 */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-blue-50">
              <TrendingUp size={20} className="text-blue-600" />
            </div>
            <span className="text-xs text-gray-400">Hit Rate</span>
          </div>
          <div className="mt-3 text-3xl font-semibold text-gray-900">{metrics.hr.toFixed(4)}</div>
          <div className="mt-1 text-xs text-gray-500">HR@1 命中率</div>
        </div>

        <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-emerald-50">
              <BarChart3 size={20} className="text-emerald-600" />
            </div>
            <span className="text-xs text-gray-400">Recall@5</span>
          </div>
          <div className="mt-3 text-3xl font-semibold text-gray-900">
            {metrics.recallAtK[5]?.toFixed(4) ?? '—'}
          </div>
          <div className="mt-1 text-xs text-gray-500">Top-5 召回率</div>
        </div>

        <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-amber-50">
              <Activity size={20} className="text-amber-600" />
            </div>
            <span className="text-xs text-gray-400">NDCG@5</span>
          </div>
          <div className="mt-3 text-3xl font-semibold text-gray-900">
            {metrics.ndcgAtK[5]?.toFixed(4) ?? '—'}
          </div>
          <div className="mt-1 text-xs text-gray-500">归一化折损累积增益</div>
        </div>

        <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-sky-50">
              <Users size={20} className="text-sky-600" />
            </div>
            <span className="text-xs text-gray-400">测试集</span>
          </div>
          <div className="mt-3 text-3xl font-semibold text-gray-900">
            {metrics.totalUsers.toLocaleString()}
          </div>
          <div className="mt-1 text-xs text-gray-500">用户数</div>
        </div>
      </div>

      {/* 训练曲线 */}
      <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
        <div className="mb-1 flex items-center justify-between">
          <div className="text-sm font-semibold text-gray-800">LoRA 微调训练曲线</div>
          <div className="text-xs text-gray-400">{logs.description}</div>
        </div>
        <div className="mb-4 flex gap-4 text-xs text-gray-500">
          <span>总步数: {logs.total_steps}</span>
          <span>最终 train_loss: {logs.steps[logs.steps.length - 1]?.loss ?? '—'}</span>
          <span>最佳 eval_loss: {logs.best_eval_loss}</span>
        </div>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={trainData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="step"
              label={{ value: 'Step', position: 'insideBottom', offset: -5 }}
              tick={{ fontSize: 12 }}
            />
            <YAxis
              domain={[0, 'auto']}
              tick={{ fontSize: 12 }}
              label={{ value: 'Loss', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#fff',
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
                fontSize: '12px',
              }}
            />
            <Legend wrapperStyle={{ fontSize: '12px' }} />
            <Line
              type="monotone"
              dataKey="loss"
              name="Train Loss"
              stroke="#6366f1"
              strokeWidth={2}
              dot={{ r: 4 }}
            />
            <Line
              type="monotone"
              dataKey="eval_loss"
              name="Eval Loss"
              stroke="#ef4444"
              strokeWidth={2}
              dot={{ r: 4 }}
              connectNulls
            />
          </LineChart>
        </ResponsiveContainer>
        <p className="mt-3 text-xs text-gray-400">
          横轴为训练步数（{logs.steps[0]?.step} ~ {logs.total_steps}），纵轴为损失值。蓝线为训练损失，红线为验证损失，整体呈收敛趋势。
        </p>
      </div>

      {/* 补充指标 + DG 基线对比 */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
          <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-gray-800">
            <Target size={16} className="text-blue-500" />
            补充指标
          </div>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-500">模糊匹配 HR@1</span>
              <span className="text-sm font-medium text-blue-600">{(extraMetrics.fuzzy_HR1 * 100).toFixed(2)}%</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-500">部分匹配 HR@1</span>
              <span className="text-sm font-medium text-gray-700">{(extraMetrics.partial_HR1 * 100).toFixed(2)}%</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-500">MRR</span>
              <span className="text-sm font-medium text-gray-700">{extraMetrics.mrr.toFixed(4)}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-500">域外率 (OOD)</span>
              <span className="text-sm font-medium text-amber-600">
                {(extraMetrics.ood_rate * 100).toFixed(2)}% ({extraMetrics.ood_count}/{extraMetrics.total})
              </span>
            </div>
          </div>
          <p className="mt-3 text-xs text-gray-400">
            模糊匹配:生成结果与真实标签有部分重叠视为命中。域外率:推荐落在源域外的比例。
          </p>
        </div>

        <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
          <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-gray-800">
            <Globe size={16} className="text-gray-400" />
            DG 基线对比
          </div>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-gray-100 text-gray-400">
                <th className="py-2 text-left font-medium">方法</th>
                <th className="py-2 text-right font-medium">HR@1</th>
                <th className="py-2 text-right font-medium">HR@10</th>
                <th className="py-2 text-right font-medium">HR@20</th>
                <th className="py-2 text-right font-medium">MRR</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-gray-50">
                <td className="py-2 text-left font-medium text-blue-600">LLM+画像</td>
                <td className="py-2 text-right">{metrics.hr.toFixed(4)}</td>
                <td className="py-2 text-right">{metrics.recallAtK[10]?.toFixed(4) ?? '—'}</td>
                <td className="py-2 text-right">{metrics.recallAtK[20]?.toFixed(4) ?? '—'}</td>
                <td className="py-2 text-right">{extraMetrics.mrr.toFixed(4)}</td>
              </tr>
              <tr>
                <td className="py-2 text-left font-medium text-gray-500">DG 基线</td>
                <td className="py-2 text-right">{extraMetrics.dg_baseline['HR@1'].toFixed(4)}</td>
                <td className="py-2 text-right">{extraMetrics.dg_baseline['HR@10'].toFixed(4)}</td>
                <td className="py-2 text-right">{extraMetrics.dg_baseline['HR@20'].toFixed(4)}</td>
                <td className="py-2 text-right">{extraMetrics.dg_baseline.MRR.toFixed(4)}</td>
              </tr>
            </tbody>
          </table>
          <p className="mt-3 text-xs text-gray-400">
            精确匹配指标偏低，反映出生成式跨域推荐任务的固有难度；模糊匹配可体现模型对用户偏好的部分捕捉能力。
          </p>
        </div>
      </div>

      {/* K 值曲线 */}
      <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
        <div className="mb-4 text-sm font-semibold text-gray-800">
          Recall@K & NDCG@K 曲线
        </div>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="k"
              label={{ value: 'K (Top-K)', position: 'insideBottom', offset: -5 }}
              tick={{ fontSize: 12 }}
            />
            <YAxis
              domain={[0, 1]}
              tick={{ fontSize: 12 }}
              label={{ value: 'Score', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#fff',
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
                fontSize: '12px',
              }}
            />
            <Legend wrapperStyle={{ fontSize: '12px' }} />
            <Line
              type="monotone"
              dataKey="Recall"
              stroke="#10b981"
              strokeWidth={2}
              dot={{ r: 4 }}
            />
            <Line
              type="monotone"
              dataKey="NDCG"
              stroke="#f59e0b"
              strokeWidth={2}
              dot={{ r: 4 }}
            />
          </LineChart>
        </ResponsiveContainer>
        <p className="mt-3 text-xs text-gray-400">
          横轴为推荐列表长度 K，纵轴为评测指标得分。Recall 衡量"真实喜欢的物品被召回的比例"，NDCG 额外考虑排序质量。
        </p>
      </div>
    </div>
  )
}
