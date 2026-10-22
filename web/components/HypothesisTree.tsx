// 내부 가설 트리(자세히 보기 안에서만). 토큰 기반 클래스만 사용.
type Node = {
  node_id: string;
  parent_id: string | null;
  level: number;
  hypothesis: string;
  sub_questions: string[];
  required_evidence: {
    evidence_type: string;
    specificity: string;
    recency_required_months: number;
  }[];
  priority: number;
  mece_rationale?: string | null;
};

function childrenOf(nodes: Node[], id: string | null) {
  return nodes.filter((n) => (n.parent_id ?? null) === id);
}

function NodeView({ node, nodes }: { node: Node; nodes: Node[] }) {
  const kids = childrenOf(nodes, node.node_id);
  return (
    <div className="tnode">
      <div className="thead">
        <span className="mono">{node.node_id}</span>
        <span className="pill-mono">p{node.priority}</span>
        <strong>{node.hypothesis}</strong>
      </div>
      {node.mece_rationale && <div className="mece">MECE: {node.mece_rationale}</div>}
      {node.sub_questions?.length > 0 && (
        <ul className="subq">
          {node.sub_questions.map((q, i) => (
            <li key={i}>{q}</li>
          ))}
        </ul>
      )}
      {node.required_evidence?.length > 0 && (
        <div className="reqev">
          required_evidence:{" "}
          {node.required_evidence
            .map((e) => `${e.evidence_type}(${e.specificity}, ${e.recency_required_months}mo)`)
            .join(" · ")}
        </div>
      )}
      {kids.map((k) => (
        <NodeView key={k.node_id} node={k} nodes={nodes} />
      ))}
    </div>
  );
}

export function HypothesisTree({ nodes }: { nodes: Node[] }) {
  return (
    <div>
      {childrenOf(nodes, null).map((r) => (
        <NodeView key={r.node_id} node={r} nodes={nodes} />
      ))}
    </div>
  );
}
