"""가설 트리 출력 스키마 (Pydantic). Claude tool-use의 input_schema로도 쓰인다.

규칙: 모든 노드에 required_evidence를 명시한다 (빼먹지 않는다).
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator

from app.schemas.coerce import loads_json_fields


class EvidenceType(str, Enum):
    """필요 근거의 종류 (MECE-ish 카테고리)."""

    market_sizing = "market_sizing"
    financial_performance = "financial_performance"
    customer_behavior = "customer_behavior"
    competitor_benchmark = "competitor_benchmark"
    operational_metrics = "operational_metrics"
    cost_structure = "cost_structure"
    pricing_data = "pricing_data"
    industry_report = "industry_report"
    regulatory = "regulatory"
    expert_interview = "expert_interview"
    survey = "survey"
    internal_document = "internal_document"
    other = "other"


class Specificity(str, Enum):
    quantitative = "quantitative"
    qualitative = "qualitative"


class RequiredEvidence(BaseModel):
    """리프(및 노드)를 데이터로 뒷받침하기 위해 필요한 근거 1건."""

    evidence_type: EvidenceType
    specificity: Specificity
    recency_required_months: int = Field(
        ge=0, le=600,
        description="근거의 최신성 요구(개월). 0이면 시점 무관(구조적/영속적 사실).",
    )
    description: str | None = Field(
        default=None, description="구체적으로 어떤 데이터/지표인지 한 줄."
    )


class HypothesisNode(BaseModel):
    """가설 트리의 노드 1개 (flat 배열 + parent_id로 트리 표현)."""

    node_id: str = Field(description="예: '1', '1.2', '1.2.3'")
    parent_id: str | None = Field(
        default=None, description="1레벨 노드는 null, 그 외는 부모 node_id."
    )
    level: int = Field(ge=1, le=4, description="1=최상단 브랜치 … 최대 4까지.")
    hypothesis: str = Field(description="검증 대상 가설(주장문).")
    sub_questions: list[str] = Field(min_length=1, description="이 노드가 답해야 할 하위 질문.")
    required_evidence: list[RequiredEvidence] = Field(
        min_length=1, description="이 노드를 뒷받침할 근거(비어 있으면 안 됨)."
    )
    priority: int = Field(ge=1, le=5, description="1(낮음)~5(높음).")
    is_leaf: bool = Field(description="데이터로 직접 답할 수 있는 리프면 true.")
    mece_rationale: str | None = Field(
        default=None,
        description="1레벨 노드만 필수: 형제 브랜치와 왜 MECE인지 한 줄.",
    )

    @model_validator(mode="before")
    @classmethod
    def _unwrap_json_strings(cls, data):
        return loads_json_fields(data, ("sub_questions", "required_evidence"))

    @model_validator(mode="after")
    def _level1_needs_mece(self) -> HypothesisNode:
        if self.level == 1 and not (self.mece_rationale and self.mece_rationale.strip()):
            raise ValueError(f"level-1 노드({self.node_id})는 mece_rationale이 필요하다.")
        if self.level == 1 and self.parent_id not in (None, ""):
            raise ValueError(f"level-1 노드({self.node_id})는 parent_id가 null이어야 한다.")
        if self.level > 1 and not self.parent_id:
            raise ValueError(f"노드({self.node_id})는 parent_id가 필요하다.")
        return self


class HypothesisTree(BaseModel):
    """set_project_brief의 성공 출력."""

    primary_question: str = Field(description="피라미드 최상단의 단일 핵심 질문.")
    hypothesis_tree: list[HypothesisNode] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _unwrap_json_strings(cls, data):
        return loads_json_fields(data, ("hypothesis_tree",))

    @model_validator(mode="after")
    def _validate_structure(self) -> HypothesisTree:
        ids = [n.node_id for n in self.hypothesis_tree]
        if len(ids) != len(set(ids)):
            raise ValueError("node_id가 중복됐다.")
        id_set = set(ids)

        level1 = [n for n in self.hypothesis_tree if n.level == 1]
        if len(level1) < 2:
            raise ValueError("MECE 분해를 위해 1레벨 브랜치가 최소 2개 필요하다.")

        for n in self.hypothesis_tree:
            if n.level > 1 and n.parent_id not in id_set:
                raise ValueError(f"노드({n.node_id})의 parent_id({n.parent_id})가 트리에 없다.")

        if not any(n.is_leaf for n in self.hypothesis_tree):
            raise ValueError("리프 노드가 최소 1개는 있어야 한다.")

        by_id = {n.node_id: n for n in self.hypothesis_tree}

        # parent_id 순환 금지 (레벨 검사보다 먼저 해야 원인이 정확히 드러난다)
        for n in self.hypothesis_tree:
            seen: list[str] = []
            cur: str | None = n.node_id
            while cur:
                if cur in seen:
                    path = " -> ".join([*seen, cur])
                    raise ValueError(f"parent_id 순환이 있다: {path}")
                seen.append(cur)
                cur = by_id[cur].parent_id or None

        # level == 부모 level + 1
        for n in self.hypothesis_tree:
            if n.parent_id:
                parent = by_id[n.parent_id]
                if n.level != parent.level + 1:
                    raise ValueError(
                        f"노드({n.node_id})의 level {n.level}이 부모({parent.node_id}) "
                        f"level {parent.level}+1과 다르다."
                    )

        # is_leaf == 자식 없음
        has_child = {n.parent_id for n in self.hypothesis_tree if n.parent_id}
        for n in self.hypothesis_tree:
            if n.is_leaf == (n.node_id in has_child):
                raise ValueError(
                    f"노드({n.node_id})의 is_leaf={n.is_leaf}인데 자식이 "
                    f"{'있다' if n.node_id in has_child else '없다'}."
                )
        return self


class ClarificationRequest(BaseModel):
    """브리프가 너무 모호할 때의 출력 (질문 최대 1개)."""

    clarification_question: str = Field(
        description="트리를 만들기 위해 반드시 필요한 단 하나의 명확화 질문."
    )
