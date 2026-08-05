/**
 * 前端演示与评测数据导入
 *   - datasets: 演示用户画像与推理结果 (datasets.json)
 *   - evalData: GM / AO 双数据集真实评估指标 (eval_data.json)
 *   - trainingLogs: GM / AO 微调训练日志 (training_logs.json)
 */
import type { Dataset, EvalDataset, EvalKey, TrainingLog } from './types'
import trainingLogsData from './data/training_logs.json'
import datasetsData from './data/datasets.json'
import evalDataRaw from './data/eval_data.json'

export const datasets = datasetsData as Dataset[]

export function findUser(datasetKey: string, userId: string) {
  const ds = datasets.find((d) => d.key === datasetKey)
  return ds?.users.find((u) => u.id === userId)
}

// GM / AO 双数据集真实评估指标
export const evalData = evalDataRaw as Record<EvalKey, EvalDataset>

// GM / AO 微调训练日志：{ GM: {...}, AO: {...} }
export const trainingLogs = trainingLogsData as Record<EvalKey, TrainingLog>
