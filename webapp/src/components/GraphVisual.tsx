import type { HistoryItem } from '../types'

const DOT_COLOR: Record<HistoryItem['kind'], string> = {
  movie: '#3b82f6', // blue
  game: '#10b981', // emerald
  art: '#f59e0b', // amber
  office: '#0ea5e9', // sky
}

// 把用户的历史交互序列画成一条"节点-边"链：
// 节点 = 交互过的物品，边 = 序列上的先后关系（简化版的用户-物品交互图）
export function GraphVisual({ history }: { history: HistoryItem[] }) {
  const nodeGap = 74
  const width = Math.max(240, (history.length - 1) * nodeGap + 40)
  const cy = 44

  return (
    <div className="overflow-x-auto">
      <svg width={width} height={88} className="min-w-full">
        {history.slice(0, -1).map((_, idx) => (
          <line
            key={idx}
            x1={20 + idx * nodeGap}
            y1={cy}
            x2={20 + (idx + 1) * nodeGap}
            y2={cy}
            stroke="#d1d5db"
            strokeWidth={2}
          />
        ))}
        {history.map((item, idx) => (
          <g key={idx}>
            <circle
              cx={20 + idx * nodeGap}
              cy={cy}
              r={14}
              fill={DOT_COLOR[item.kind]}
              opacity={0.15}
            />
            <circle
              cx={20 + idx * nodeGap}
              cy={cy}
              r={7}
              fill={DOT_COLOR[item.kind]}
            />
            <text
              x={20 + idx * nodeGap}
              y={cy + 30}
              textAnchor="middle"
              fontSize={10}
              fill="#6b7280"
            >
              {idx + 1}
            </text>
          </g>
        ))}
      </svg>
    </div>
  )
}
