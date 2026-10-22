const META: Record<string, { icon: string; label: string }> = {
  SUPPORTS: { icon: "✓", label: "지지" },
  REFUTES: { icon: "✗", label: "반증" },
  WEAK_RELATED: { icon: "~", label: "약한 관련" },
  IRRELEVANT: { icon: "·", label: "무관" },
};

export function StanceBadge({ stance }: { stance: string }) {
  const m = META[stance] ?? { icon: "?", label: stance };
  return (
    <span className={`badge ${stance}`}>
      {m.icon} {m.label}
    </span>
  );
}
