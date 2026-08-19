export type ItemKind = 'movie' | 'game' | 'art' | 'office'

export interface HistoryItem {
  title: string
  kind: ItemKind
}

export interface SimilarUser {
  id: string
  similarity: number
  recentTitles: string[]
}

export interface RecommendationResult {
  title: string
  kind: ItemKind
  reason: string
  similarUsers: SimilarUser[]
}

// 检索候选池：双图对比学习编码后，按相似度排序得到的候选用户全量列表
// (只有 selected=true 的前 K 个会被拼进最终喂给 LLM 的 prompt)
export interface RetrievalCandidate {
  id: string
  similarity: number
  recentTitles: string[]
  selected: boolean
}

export interface DemoUser {
  id: string
  history: HistoryItem[]
  result: RecommendationResult
  retrievalPool: RetrievalCandidate[]
}

export type DatasetKey = 'GM'

// 评测数据集 key：GM / AO 两个跨域推荐数据集
export type EvalKey = 'GM' | 'AO'

export interface EvalStats {
  numUsers: number
  testUsers: number
  sourceItems: number
  targetItems: number
  sourceInteractions?: number
  targetInteractions?: number
  totalInteractions?: number
  avgSeqLength: number
  trainInstructions: number
}

export interface EvalMetrics {
  hr1: number
  hr5: number
  hr10: number
  hr20: number
  ndcg1: number
  ndcg5: number
  ndcg10: number
  ndcg20: number
  mrr: number
  fuzzyHr1: number
  partialHr1: number
  exactHr1: number
  oodRate: number
  oodCount: number
  totalUsers: number
  coldUsers: number
  warmUsers: number
}

export interface DgBaseline {
  hr1: number
  hr5: number
  hr10: number
  hr20: number
  mrr: number
}

export interface EvalTraining {
  totalSteps: number
  epochs: number
  finalLoss: number
  bestEvalLoss: number
}

export interface ExpandedMetrics {
  hr1: number
  hr5: number
  hr10: number
  hr20: number
  ndcg1: number
  ndcg5: number
  ndcg10: number
  ndcg20: number
  mrr: number
}

export interface EvalDataset {
  label: string
  sourceDomain: string
  targetDomain: string
  stats: EvalStats
  metrics: EvalMetrics
  dgBaseline: DgBaseline
  training: EvalTraining
  expandedMetrics?: ExpandedMetrics
}

export interface TrainingStep {
  step: number
  loss: number
  learning_rate: number
  epoch: number
  eval_loss?: number
}

export interface TrainingLog {
  steps: TrainingStep[]
  description: string
  total_steps: number
  best_eval_loss: number
}

export interface Dataset {
  key: DatasetKey
  label: string
  users: DemoUser[]
}

export interface MetricsSnapshot {
  hr: number
  recallAtK: Record<number, number> // k -> recall value
  ndcgAtK: Record<number, number>   // k -> ndcg value
  totalUsers: number
  meanSimilarUserCount: number
}
