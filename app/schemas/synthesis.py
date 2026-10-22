"""find_synthesis_candidates 출력 스키마.

제안은 마법이 아니라 '노드별 증거요건 충족 매트릭스'의 조합 탐색 결과이며, 각 제안에는
왜 이 연결이 성립하는지(warrant)가 반드시 붙는다. 근거 없는 제안은 게이트에서 걸러진다.
"""
from pydantic import BaseModel, Field


class SynthesisLinkCheck(BaseModel):
    """Claude 게이트가 채우는 단일 (자료→노드×증거유형) 연결 판정."""

    connected: bool = Field(description="자료가 이 노드의 지정 증거유형에 실질적으로 기여하는가.")
    warrant: str = Field(description="왜 연결이 성립/불성립하는지의 구체적 근거.")
    confidence: float = Field(ge=0, le=1)


class SynthesisProposal(BaseModel):
    kind: str = Field(description="fill_gap | combine | weak_promote")
    target_node_id: str
    target_hypothesis: str
    content_ids: list[str] = Field(description="엮을 자료 id(들). combine이면 2개 이상.")
    fills_evidence_types: list[str] = Field(description="이 제안으로 채워지는 증거 유형.")
    warrant: str
    confidence: float


class SynthesisResult(BaseModel):
    project_id: str
    considered_links: int = Field(description="결정론적으로 열거된 후보 연결 수(게이트 이전).")
    dropped_by_warrant_gate: int = Field(description="근거 불충분으로 걸러진 수.")
    truncated_candidates: int = Field(default=0, description="상한 초과로 평가 못 한 후보 수(무음 절단 방지).")
    proposals: list[SynthesisProposal]
