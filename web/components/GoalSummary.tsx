// 목표 요약: 핵심 질문 + 1레벨 목표만 '사람 말'로.
// 내부 필드(node_id/priority/required_evidence/MECE)는 여기서 보여주지 않는다.
type Node = { node_id: string; level: number; hypothesis: string };

const MARKS = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩"];

export function GoalSummary({
  primaryQuestion,
  nodes,
}: {
  primaryQuestion: string;
  nodes: Node[];
}) {
  const goals = nodes.filter((n) => n.level === 1);
  return (
    <div>
      <span className="label">이 프로젝트의 목표</span>
      <div className="goal-q">{primaryQuestion}</div>
      <span className="label">세부 목표</span>
      <ul className="goal-list">
        {goals.map((g, i) => (
          <li key={g.node_id}>
            <span className="gmark">{MARKS[i] ?? "•"}</span>
            <span className="gtext">{g.hypothesis}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
