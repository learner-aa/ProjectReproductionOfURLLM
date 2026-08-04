/**
 * 前端数据加载工具
 * 从 public/data/ 目录读取预处理好的JSON数据
 */

export interface RetrievalMatrix {
  matrix: number[][]
  user_indices: number[]
  shape: number[]
  description: string
}

export interface TestCandidates {
  candidates: number[][]
  user_indices: number[]
  shape: number[]
  description: string
}

export interface UserFeatures {
  train: {
    num_users: number
    source_domain_dim: number
    target_domain_dim: number
    source_mean: number
    source_std: number
    target_mean: number
    target_std: number
  }
  test: {
    num_users: number
    source_domain_dim: number
    target_domain_dim: number
    source_mean: number
    source_std: number
    target_mean: number
    target_std: number
  }
  description: string
}

export interface TrainingLog {
  epoch: number
  train_loss: number
  ce_loss: number
  mi_loss: number
  al_loss: number
  val_X_MRR: number
  val_Y_MRR: number
  val_Y_HR10: number
}

export interface TrainingLogs {
  epochs: TrainingLog[]
  best_epoch: number
  description: string
}

export interface DatasetStats {
  movie_game: {
    name: string
    source_domain: { name: string; num_items: number; num_interactions: number }
    target_domain: { name: string; num_items: number; num_interactions: number }
    num_users: number
    avg_sequence_length: number
  }
  art_office: {
    name: string
    source_domain: { name: string; num_items: number; num_interactions: number }
    target_domain: { name: string; num_items: number; num_interactions: number }
    num_users: number
    avg_sequence_length: number
  }
}

/**
 * 通用JSON加载函数
 */
async function fetchJSON<T>(path: string): Promise<T | null> {
  try {
    const response = await fetch(path)
    if (!response.ok) {
      console.warn(`Failed to fetch ${path}: ${response.statusText}`)
      return null
    }
    return await response.json()
  } catch (error) {
    console.error(`Error loading ${path}:`, error)
    return null
  }
}

/**
 * 加载检索矩阵（采样后的）
 */
export async function loadRetrievalMatrix(): Promise<RetrievalMatrix | null> {
  return fetchJSON<RetrievalMatrix>('/data/retrieval_matrix_sample.json')
}

/**
 * 加载推荐候选
 */
export async function loadTestCandidates(): Promise<TestCandidates | null> {
  return fetchJSON<TestCandidates>('/data/test_candidates_sample.json')
}

/**
 * 加载用户特征统计
 */
export async function loadUserFeatures(): Promise<UserFeatures | null> {
  return fetchJSON<UserFeatures>('/data/user_features.json')
}

/**
 * 加载训练日志
 */
export async function loadTrainingLogs(): Promise<TrainingLogs | null> {
  return fetchJSON<TrainingLogs>('/data/training_logs.json')
}

/**
 * 加载数据集统计
 */
export async function loadDatasetStats(): Promise<DatasetStats | null> {
  return fetchJSON<DatasetStats>('/data/dataset_stats.json')
}

/**
 * 预加载所有数据（可在App启动时调用）
 */
export async function preloadAllData() {
  console.log('🔄 预加载数据...')

  const [
    retrievalMatrix,
    testCandidates,
    userFeatures,
    trainingLogs,
    datasetStats
  ] = await Promise.all([
    loadRetrievalMatrix(),
    loadTestCandidates(),
    loadUserFeatures(),
    loadTrainingLogs(),
    loadDatasetStats()
  ])

  console.log('✅ 数据加载完成:', {
    retrievalMatrix: retrievalMatrix ? '✓' : '✗',
    testCandidates: testCandidates ? '✓' : '✗',
    userFeatures: userFeatures ? '✓' : '✗',
    trainingLogs: trainingLogs ? '✓' : '✗',
    datasetStats: datasetStats ? '✓' : '✗'
  })

  return {
    retrievalMatrix,
    testCandidates,
    userFeatures,
    trainingLogs,
    datasetStats
  }
}

/**
 * 根据用户ID获取相似用户（从检索矩阵）
 */
export function getSimilarUsers(
  matrix: RetrievalMatrix,
  userIndex: number,
  topK: number = 10
): Array<{ userId: number; similarity: number }> {
  if (!matrix || userIndex >= matrix.matrix.length) {
    return []
  }

  const similarities = matrix.matrix[userIndex]
  const indexed = similarities.map((sim, idx) => ({ userId: idx, similarity: sim }))

  // 按相似度降序排序
  indexed.sort((a, b) => b.similarity - a.similarity)

  return indexed.slice(0, topK)
}

/**
 * 根据用户ID获取推荐候选
 */
export function getRecommendationCandidates(
  candidates: TestCandidates,
  userIndex: number,
  topK: number = 20
): number[] {
  if (!candidates || userIndex >= candidates.candidates.length) {
    return []
  }

  return candidates.candidates[userIndex].slice(0, topK)
}
