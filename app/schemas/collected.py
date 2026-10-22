"""list_collected 출력 스키마 — 목표 충족 현황 뷰.

check_goal_drift의 입력이 되므로 "채워진 목표 / 빈 목표 / 미분류 자료"를 명확히
구조화한다.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class EvidenceCoverage(BaseModel):
    """노드의 required_evidence 항목 1개의 충족 여부."""

    evidence_type: str
    specificity: str
    recency_required_months: int
    covered: bool
    provided_by: list[str] = Field(
        default_factory=list, description="이 유형을 제공한 content_id 목록."
    )


class NodeStatus(BaseModel):
    node_id: str
    parent_id: str | None = None
    hypothesis: str
    level: int
    priority: int
    supports_count: int
    refutes_count: int
    weak_related_count: int = 0
    supporting_content_ids: list[str] = Field(default_factory=list)
    refuting_content_ids: list[str] = Field(default_factory=list)
    weak_related_content_ids: list[str] = Field(
        default_factory=list, description="이 노드와 약하게 관련된(증거 미확정) 자료 — synthesis 후보."
    )
    max_support_strength: float = 0.0
    required_evidence_total: int = 0
    required_evidence_covered: int = 0
    evidence_coverage: list[EvidenceCoverage] = Field(default_factory=list)


class GapNode(BaseModel):
    node_id: str
    hypothesis: str
    priority: int


class UnclassifiedItem(BaseModel):
    """어떤 노드에도 SUPPORTS/REFUTES로 걸리지 못한 자료.

    category: 'WEAK_RELATED'(약한 관련 노드가 하나라도 있음 — synthesis 후보) 또는
    'IRRELEVANT'(전부 무관).
    """

    content_id: str
    source_url: str | None = None
    preview: str
    category: str
    best_node_id: str | None = None
    best_similarity: float = 0.0
    weak_related_node_ids: list[str] = Field(default_factory=list)


class CollectedSummary(BaseModel):
    total_nodes: int
    nodes_with_support: int
    nodes_with_any_evidence: int
    coverage_ratio: float = Field(description="지지 증거 1건 이상 보유 노드 / 전체 노드.")
    total_content_items: int
    classified_items: int
    unclassified_items: int
    weak_related_items: int
    irrelevant_items: int


class CollectedView(BaseModel):
    project_id: str
    primary_question: str
    summary: CollectedSummary
    nodes: list[NodeStatus]
    gap_nodes: list[GapNode] = Field(
        default_factory=list, description="증거(지지·반증) 0건 노드."
    )
    # 미분류 자료를 두 갈래로 분리해 노출한다.
    weak_related: list[UnclassifiedItem] = Field(
        default_factory=list, description="약한 관련 자료 — synthesis가 엮을 후보."
    )
    irrelevant: list[UnclassifiedItem] = Field(
        default_factory=list, description="무관 자료 — 어느 목표와도 관련 없음."
    )
