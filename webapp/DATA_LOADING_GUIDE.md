# 前端数据加载使用指南

## 📖 概述

本指南说明如何在URLLM前端中加载和使用真实数据。

---

## 🔄 数据转换流程

### 步骤1: 运行Python脚本转换数据

```bash
# 在项目根目录执行
python scripts/convert_npy_to_json.py
```

**输出文件** (位于 `frontend/public/data/`):
- `retrieval_matrix_sample.json` - 采样后的检索矩阵 (100个用户)
- `test_candidates_sample.json` - 推荐候选 (100个用户)
- `user_features.json` - 用户特征统计信息
- `training_logs.json` - 训练日志数据
- `dataset_stats.json` - 数据集统计信息

### 步骤2: 前端读取数据

前端通过 `src/utils/dataLoader.ts` 提供的函数读取数据：

```typescript
import { loadDatasetStats, loadUserFeatures, loadTrainingLogs } from '@/utils/dataLoader'

// 在组件中使用
const [stats, setStats] = useState(null)

useEffect(() => {
  async function loadData() {
    const data = await loadDatasetStats()
    setStats(data)
  }
  loadData()
}, [])
```

---

## 📦 数据加载API

### 1. 加载数据集统计

```typescript
const stats = await loadDatasetStats()

// 返回格式:
{
  movie_game: {
    name: "Entertainment-Education (Movie-Game)",
    source_domain: { name: "Movies", num_items: 18396, num_interactions: 156842 },
    target_domain: { name: "Games", num_items: 17545, num_interactions: 89653 },
    num_users: 12847,
    avg_sequence_length: 8.3
  },
  art_office: { ... }
}
```

### 2. 加载用户特征统计

```typescript
const features = await loadUserFeatures()

// 返回格式:
{
  train: {
    num_users: 35941,
    source_domain_dim: 328,
    target_domain_dim: 328,
    source_mean: 0.0234,
    source_std: 0.1567
  },
  test: { ... }
}
```

### 3. 加载训练日志

```typescript
const logs = await loadTrainingLogs()

// 返回格式:
{
  epochs: [
    { epoch: 1, train_loss: 10.32, val_Y_MRR: 0.000355, ... },
    { epoch: 2, train_loss: 8.59, val_Y_MRR: 0.000397, ... },
    ...
  ],
  best_epoch: 17
}
```

### 4. 加载检索矩阵

```typescript
const matrix = await loadRetrievalMatrix()

// 返回格式:
{
  matrix: [[0.91, 0.87, ...], ...],  // 100x35941 的相似度矩阵
  user_indices: [0, 15, 23, ...],     // 采样的用户索引
  shape: [100, 35941]
}
```

### 5. 加载推荐候选

```typescript
const candidates = await loadTestCandidates()

// 返回格式:
{
  candidates: [[1234, 5678, ...], ...],  // 每个用户的Top-10000候选物品ID
  user_indices: [0, 15, 23, ...],
  shape: [100, 10000]
}
```

---

## 🎯 实际使用示例

### 示例1: 数据集概览页面

已实现在 `src/pages/DatasetOverview.tsx`:

```typescript
export function DatasetOverview() {
  const [stats, setStats] = useState<DatasetStats | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function loadData() {
      setLoading(true)
      const data = await loadDatasetStats()
      setStats(data)
      setLoading(false)
    }
    loadData()
  }, [])

  if (loading) return <div>加载中...</div>
  if (!stats) return <div>加载失败</div>

  return (
    <div>
      <h2>{stats.movie_game.name}</h2>
      <p>用户数: {stats.movie_game.num_users}</p>
    </div>
  )
}
```

### 示例2: 获取用户的相似用户

```typescript
import { loadRetrievalMatrix, getSimilarUsers } from '@/utils/dataLoader'

const matrix = await loadRetrievalMatrix()
const similarUsers = getSimilarUsers(matrix, 0, 10)  // 获取用户0的Top-10相似用户

// 返回:
// [
//   { userId: 889, similarity: 0.91 },
//   { userId: 341, similarity: 0.87 },
//   ...
// ]
```

### 示例3: 获取推荐候选

```typescript
import { loadTestCandidates, getRecommendationCandidates } from '@/utils/dataLoader'

const candidates = await loadTestCandidates()
const topCandidates = getRecommendationCandidates(candidates, 0, 20)

// 返回: [1234, 5678, 9012, ...]  物品ID列表
```

---

## 🚀 集成到现有页面

### 更新 App.tsx 添加新页面

```typescript
import { DatasetOverview } from './pages/DatasetOverview'

const NAV = [
  { key: 'workbench', label: '推荐工作台' },
  { key: 'retrieval', label: '相似用户检索' },
  { key: 'overview', label: '数据集概览' },  // 新增
]

// 在 main 中添加路由
{page === 'overview' && <DatasetOverview />}
```

---

## ⚠️ 注意事项

### 1. 数据文件大小限制

由于完整的检索矩阵有494MB，我们只采样了100个用户用于展示。如果需要完整数据：

```python
# 在 convert_npy_to_json.py 中修改
sample_size=3601  # 使用全部用户（但会导致JSON文件很大）
```

### 2. 如果数据文件不存在

前端会自动降级到 mock 数据，不会报错：

```typescript
const stats = await loadDatasetStats()
if (!stats) {
  // 使用 mockData.ts 中的数据
  return mockDatasets
}
```

### 3. CORS问题

如果遇到跨域问题，确保 `public/data/` 目录存在，Vite会自动处理静态文件服务。

---

## 🔌 下一步：接入后端API

当前方案是**静态数据展示**，无法调用真实的LLaMA2模型。如果要实现实时推理：

### 方案A: 简单的Flask后端

```python
# backend/app.py
from flask import Flask, jsonify, request
import numpy as np

app = Flask(__name__)

# 加载数据
retrieval_matrix = np.load('best_trte_XORY_DG_.npy')

@app.route('/api/recommend', methods=['POST'])
def recommend():
    user_id = request.json['user_id']
    # 调用 LLaMA2 推理
    result = generate_recommendation(user_id)
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000)
```

### 方案B: FastAPI后端（推荐）

```python
# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

@app.post("/api/recommend")
async def recommend(user_id: str, dataset: str):
    # 1. 加载用户特征
    # 2. 检索相似用户
    # 3. 构建 LLM prompt
    # 4. 调用 LLaMA2
    return {"result": {...}}
```

前端调用：

```typescript
// src/api/client.ts
export async function callRecommendAPI(userId: string, dataset: string) {
  const response = await fetch('http://localhost:5000/api/recommend', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: userId, dataset })
  })
  return response.json()
}
```

---

## 📝 总结

**当前方案（推荐）**：
- ✅ 运行 Python 脚本转换数据为 JSON
- ✅ 前端直接 fetch 读取静态 JSON
- ✅ 无需后端服务器
- ✅ 适合演示和原型开发

**未来方案**：
- 🔄 搭建 FastAPI 后端
- 🔄 接入真实 LLaMA2 推理
- 🔄 支持实时生成推荐
