# URLLM 前端开发计划

## 已完成功能 ✅
- [x] 推荐工作台页面
- [x] 相似用户检索页面
- [x] 用户历史交互展示
- [x] 交互图可视化
- [x] 检索候选池排行
- [x] Mock数据演示

## 短期优化 (1-2天)

### 1. 数据对接
- [ ] 编写 Python 脚本将 npy 文件转为 JSON
  - `best_trte_XORY_DG_.npy` → `retrieval_matrix.json`
  - `t4_G2_final_DGresult_test_candidate.npy` → `test_candidates.json`
  - `saved_models/*.npy` → `user_features.json`

### 2. 新增页面
- [ ] 数据集概览页
  - 数据集统计信息
  - 源域/目标域分布图表
  - 交互序列长度分布
  
- [ ] 模型性能页
  - 从 `logs.txt` 读取训练日志
  - 绘制训练曲线 (Loss, MRR, HR@10)
  - 展示最佳epoch指标

- [ ] Pipeline流程图页
  - 可视化端到端pipeline
  - 标注每个阶段的文件路径
  - 显示当前进度状态

### 3. 交互增强
- [ ] 添加推荐结果对比功能 (不同用户的推荐结果并排显示)
- [ ] 支持导出推荐结果为 PDF/Excel
- [ ] 添加搜索功能 (按用户ID快速定位)

## 中期开发 (3-5天)

### 4. 后端API开发
```python
# 使用 FastAPI 搭建推理服务
# backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class RecommendRequest(BaseModel):
    user_id: str
    dataset_key: str

@app.post("/api/recommend")
async def recommend(req: RecommendRequest):
    # 调用 enhancement/src/llm_inference.py (LLMRecommender) 生成推荐
    # 返回推荐结果
    pass

@app.get("/api/retrieval/{user_id}")
async def get_retrieval_pool(user_id: str):
    # 从 best_trte_XORY_DG_.npy 读取相似用户
    pass
```

### 5. 前端接入真实API
- [ ] 创建 `src/api/client.ts` (axios/fetch封装)
- [ ] 替换 mockData 为真实API调用
- [ ] 添加 loading 状态和错误处理
- [ ] 实现结果缓存 (React Query)

## 长期规划 (可选)

### 6. 高级功能
- [ ] A/B测试对比 (双图模型 vs 基线模型)
- [ ] 实时推理进度条 (WebSocket)
- [ ] 用户行为埋点统计
- [ ] 推荐结果质量评估工具

### 7. 部署优化
- [ ] Docker容器化
- [ ] Nginx反向代理
- [ ] 配置HTTPS
- [ ] 性能监控 (Sentry)

## 技术债务
- [ ] 添加单元测试 (Vitest)
- [ ] E2E测试 (Playwright)
- [ ] 代码质量检查 (ESLint严格模式)
- [ ] 响应式设计优化 (移动端适配)
