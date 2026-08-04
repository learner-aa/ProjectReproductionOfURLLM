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

export type DatasetKey = 'movie-game'

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
