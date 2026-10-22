"""check_goal_drift 출력 스키마.

판정 근거(어느 노드가 왜 gap인지, 어느 브랜치가 비었는지)는 결정론적 데이터에서 나오고,
narrative만 LLM이 쓴다. 확증편향 방지: REFUTES도 목표 기여로 카운트된다.
"""
from pydantic import BaseModel, Field


class DriftSignal(BaseModel):
    drifting: bool
    severity: str = Field(description="none | mild | moderate | high")
    drift_ratio: float = Field(description="(약한관련+무관) / 전체 자료.")
    irrelevant_ratio: float = Field(description="무관 / 전체 자료.")
    threshold: float
    total_items: int
    on_goal_items: int = Field(description="SUPPORTS 또는 REFUTES로 목표에 기여한 자료 수.")
    supports_items: int
    refutes_items: int = Field(description="반증으로 기여한 자료 수(정당한 진전).")
    weak_related_items: int
    irrelevant_items: int


class NeglectedGoal(BaseModel):
    node_id: str
    hypothesis: str
    priority: int
    branch: str = Field(description="소속 1레벨 브랜치 node_id.")


class BranchStatus(BaseModel):
    branch_node_id: str
    branch_hypothesis: str
    priority: int
    has_evidence: bool
    evidence_node_count: int
    total_nodes: int


class DriftNarrative(BaseModel):
    """LLM이 채우는 서술 문안."""

    narrative: str


class GoalDriftResult(BaseModel):
    project_id: str
    signal: DriftSignal
    neglected_goals: list[NeglectedGoal] = Field(description="증거 0건 노드, priority 높은 순.")
    branch_imbalance: list[BranchStatus]
    wholly_empty_branches: list[str] = Field(description="통째로 비어있는 1레벨 브랜치 node_id.")
    next_directions: list[str] = Field(description="다음 리서치 방향 제안(결정론).")
    narrative: str
