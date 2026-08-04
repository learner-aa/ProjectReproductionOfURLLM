import type { ItemKind } from '../types'
import { Clapperboard, Gamepad2, Palette, Briefcase } from 'lucide-react'

const CONFIG: Record<ItemKind, { label: string; icon: typeof Clapperboard; className: string }> = {
  movie: { label: 'Movie', icon: Clapperboard, className: 'bg-blue-50 text-blue-600 ring-blue-200' },
  game: { label: 'Game', icon: Gamepad2, className: 'bg-emerald-50 text-emerald-600 ring-emerald-200' },
  art: { label: 'Art', icon: Palette, className: 'bg-amber-50 text-amber-600 ring-amber-200' },
  office: { label: 'Office', icon: Briefcase, className: 'bg-sky-50 text-sky-600 ring-sky-200' },
}

export function KindBadge({ kind }: { kind: ItemKind }) {
  const { label, icon: Icon, className } = CONFIG[kind]
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${className}`}
    >
      <Icon size={12} strokeWidth={2.5} />
      {label}
    </span>
  )
}
