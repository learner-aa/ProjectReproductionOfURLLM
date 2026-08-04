/**
 * 前端演示数据 - 来自 enhancement pipeline 真实产物
 * 数据源:
 *   - datasets: test_instructions.json + test_predictions.json + 物品 Jaccard 相似度
 *   - metricsSnapshots: evaluation.json 真实评估指标
 *   - extraMetrics: fuzzy_HR@1, OOD rate, DG 基线 等补充指标
 * 生成脚本: scripts/generate_data.py
 */
import type { Dataset, MetricsSnapshot } from './types'
import trainingLogsData from './data/training_logs.json'
import datasetsData from './data/datasets.json'
import realData from './data/real_data.json'

export const datasets = datasetsData as Dataset[]

export function findUser(datasetKey: string, userId: string) {
  const ds = datasets.find((d) => d.key === datasetKey)
  return ds?.users.find((u) => u.id === userId)
}

// 真实评估指标 (enhancement/outputs/eval_results/evaluation.json)
export const metricsSnapshots = realData.metricsSnapshots as Record<string, MetricsSnapshot>

// 补充指标:模糊匹配、域外率、DG 基线对比
export const extraMetrics = realData.extraMetrics

// 真实训练日志 (enhancement/outputs/lora_weights/llama2_final/trainer_state.json)
export const trainingLogs = trainingLogsData
