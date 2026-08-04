import datasetStats from '../data/dataset_stats.json'
import { Database, Cpu, GitBranch, TrendingUp, Users, Package, FileText, CheckCircle2 } from 'lucide-react'

const PIPELINE_STAGES = [
  { name: '数据预处理', desc: 'GM 数据集解析为 train/valid/test', icon: Database },
  { name: '属性提取', desc: 'DeepSeek API 提取物品属性', icon: Package },
  { name: '用户画像', desc: '行为特征 + 语义偏好画像', icon: Users },
  { name: '指令构建', desc: 'Alpaca 格式训练指令', icon: GitBranch },
  { name: 'LoRA 微调', desc: 'Qwen2-1.5B + LoRA (500 steps)', icon: Cpu },
  { name: '推理', desc: '批量推理生成推荐', icon: TrendingUp },
  { name: '评估', desc: 'HR@K / NDCG@K / MRR', icon: FileText },
]

export function Overview() {
  const stats = (datasetStats as any).movie_game

  return (
    <div className="space-y-6">
      {/* 项目简介 */}
      <div className="rounded-2xl border border-gray-200 bg-gradient-to-br from-blue-50 to-white p-6 shadow-sm">
        <h1 className="text-2xl font-bold text-gray-900">URLLM 跨域序列推荐</h1>
        <p className="mt-2 text-sm leading-relaxed text-gray-600">
          基于 LLM + 用户画像增强的跨域序列推荐系统。使用 Qwen2-1.5B-Instruct 作为基座模型，
          通过 LoRA 微调学习用户偏好模式，实现从源域(Entertainment)到目标域(Education)的跨域推荐。
          Pipeline 包含数据预处理、属性提取、用户画像、指令构建、LoRA 微调、推理、评估 7 个阶段。
        </p>
        <div className="mt-4 flex flex-wrap gap-2">
          <span className="rounded-lg bg-blue-100 px-3 py-1 text-xs font-medium text-blue-700">Qwen2-1.5B-Instruct</span>
          <span className="rounded-lg bg-emerald-100 px-3 py-1 text-xs font-medium text-emerald-700">LoRA r=8</span>
          <span className="rounded-lg bg-amber-100 px-3 py-1 text-xs font-medium text-amber-700">500 steps</span>
          <span className="rounded-lg bg-sky-100 px-3 py-1 text-xs font-medium text-sky-700">GM Dataset</span>
        </div>
      </div>

      {/* Pipeline 流程 */}
      <div className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm">
        <h2 className="mb-4 text-sm font-semibold text-gray-800">增强 Pipeline 流程 (7 Stages)</h2>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {PIPELINE_STAGES.map((stage, idx) => {
            const Icon = stage.icon
            return (
              <div key={idx} className="rounded-xl border border-gray-100 bg-gray-50/60 p-4 transition hover:border-blue-200 hover:bg-blue-50/30">
                <div className="flex items-center justify-between">
                  <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-100 text-blue-600">
                    <Icon size={16} />
                  </div>
                  <span className="flex items-center gap-1 text-xs font-medium text-emerald-600">
                    <CheckCircle2 size={12} />
                    Stage {idx + 1}
                  </span>
                </div>
                <div className="mt-2 text-sm font-semibold text-gray-800">{stage.name}</div>
                <div className="mt-1 text-xs text-gray-500">{stage.desc}</div>
              </div>
            )
          })}
        </div>
      </div>

      {/* 数据集统计 */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
          <div className="flex items-center gap-2 text-xs text-gray-400">
            <Users size={14} />
            用户总数
          </div>
          <div className="mt-1 text-2xl font-bold text-gray-900">{stats.num_users.toLocaleString()}</div>
          <div className="mt-1 text-xs text-gray-400">平均序列长度: {stats.avg_sequence_length}</div>
        </div>
        <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
          <div className="flex items-center gap-2 text-xs text-gray-400">
            <Package size={14} />
            源域 (Entertainment)
          </div>
          <div className="mt-1 text-2xl font-bold text-gray-900">{stats.source_domain.num_items.toLocaleString()}</div>
          <div className="mt-1 text-xs text-gray-400">{stats.source_domain.num_interactions.toLocaleString()} 次交互</div>
        </div>
        <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
          <div className="flex items-center gap-2 text-xs text-gray-400">
            <Package size={14} />
            目标域 (Education)
          </div>
          <div className="mt-1 text-2xl font-bold text-gray-900">{stats.target_domain.num_items.toLocaleString()}</div>
          <div className="mt-1 text-xs text-gray-400">{stats.target_domain.num_interactions.toLocaleString()} 次交互</div>
        </div>
      </div>

      {/* 关键数字 */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <div className="rounded-2xl border border-gray-200 bg-white p-5 text-center shadow-sm">
          <div className="text-2xl font-bold text-blue-600">31,570</div>
          <div className="mt-1 text-xs text-gray-500">训练指令</div>
        </div>
        <div className="rounded-2xl border border-gray-200 bg-white p-5 text-center shadow-sm">
          <div className="text-2xl font-bold text-emerald-600">3,601</div>
          <div className="mt-1 text-xs text-gray-500">测试样本</div>
        </div>
        <div className="rounded-2xl border border-gray-200 bg-white p-5 text-center shadow-sm">
          <div className="text-2xl font-bold text-amber-600">1,089,536</div>
          <div className="mt-1 text-xs text-gray-500">可训练参数</div>
        </div>
        <div className="rounded-2xl border border-gray-200 bg-white p-5 text-center shadow-sm">
          <div className="text-2xl font-bold text-sky-600">0.0705%</div>
          <div className="mt-1 text-xs text-gray-500">参数占比</div>
        </div>
      </div>
    </div>
  )
}
