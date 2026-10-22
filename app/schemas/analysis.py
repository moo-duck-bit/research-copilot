"""analyze_content 출력 스키마 (Pydantic). 스탠스 분류 결과는 Claude tool-use로 강제된다.

제시 원칙: 낙관/비관 이중서사 금지 → 지지/반증/강도 3요소로만.
"""
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from app.schemas.coerce import loads_json_fields
from app.schemas.hypothesis_tree import EvidenceType


class Stance(str, Enum):
    SUPPORTS = "SUPPORTS"
    REFUTES = "REFUTES"
    # 주제·맥락은 관련되나 지지/반증을 확정할 증거가 부족 (synthesis 후보)
    WEAK_RELATED = "WEAK_RELATED"
    # 이 가설과 실질적 관련 없음
    IRRELEVANT = "IRRELEVANT"


class StrengthFactors(BaseModel):
    """근거 강도 분해 요소. 각 0~1.

    자료에 해당 요인을 판단할 정보가 없으면 중립값/0으로 채우지 말고 null(미평가)로 둔다.
    미평가 요인은 점수 계산에서 제외되고 남은 요인만으로 재정규화된다.
    """

    evidence_type_fit: float | None = Field(
        default=None, ge=0, le=1,
        description="자료의 증거 유형이 노드의 required_evidence에 부합하는 정도. 판단 불가면 null.",
    )
    source_credibility: float | None = Field(
        default=None, ge=0, le=1,
        description="출처 신뢰도(원문에 드러난 출처/방법론 단서 기반). 단서 없으면 null.",
    )
    recency: float | None = Field(
        default=None, ge=0, le=1,
        description="최신성(노드가 요구하는 recency 대비). 시점 단서 없으면 null.",
    )
    scale: float | None = Field(
        default=None, ge=0, le=1,
        description="규모/표본(수치의 표본 크기·시장 규모 등). 규모 단서 없으면 null.",
    )


class EvidenceStrength(BaseModel):
    score: float = Field(ge=0, le=1, description="평가된 factors만의 가중합(서버가 재정규화 계산).")
    factors: StrengthFactors
    not_assessed: list[str] = Field(
        default_factory=list, description="자료에 정보가 없어 미평가(null)된 요인 이름 목록."
    )
    # 화면에는 소수점 점수 대신 등급을 앞에 둔다. 점수가 확률처럼 읽히는 것을 막기 위해서다.
    band: str = Field(default="없음", description="강 / 중 / 약 / 없음")
    assessed_count: int = Field(default=0, description="평가된 요인 수(0~4).")


# --- Claude tool-use가 채우는 입력 스키마 (노드 1개당 1회) ---
class StanceJudgment(BaseModel):
    """(노드, 자료) 쌍에 대한 스탠스 판정. Claude가 tool-use로 채운다."""

    stance: Stance
    rationale_sentences: list[str] = Field(
        default_factory=list,
        description="자료 원문에서 그대로 발췌한 문장만(요약·의역 금지). "
        "SUPPORTS/REFUTES일 때만 채우고, WEAK_RELATED/IRRELEVANT면 반드시 빈 배열.",
    )
    warrant: str = Field(
        description="이 근거가 왜 이 가설을 지지/반증하는지의 논리적 연결. WEAK_RELATED/IRRELEVANT면 왜 그렇게 판정했는지."
    )
    provided_evidence_types: list[EvidenceType] = Field(
        default_factory=list,
        description="이 자료가 실제로 제공하는 근거 유형(있는 것만). "
        "SUPPORTS/REFUTES일 때만 채우고 WEAK_RELATED/IRRELEVANT면 빈 배열.",
    )
    factors: StrengthFactors

    @model_validator(mode="before")
    @classmethod
    def _unwrap_json_strings(cls, data):
        return loads_json_fields(
            data, ("rationale_sentences", "provided_evidence_types", "factors")
        )


# --- 최종 출력 ---
class MatchedNode(BaseModel):
    node_id: str
    hypothesis: str
    candidate_similarity: float = Field(
        description="1단계 임베딩 후보 필터의 코사인 유사도(주제 근접도, 판정과 별개)."
    )
    stance: Stance
    rationale_sentences: list[str] = Field(
        description="검증된(원문에 실재하는) 발췌 문장만."
    )
    warrant: str
    provided_evidence_types: list[str] = Field(
        default_factory=list,
        description="이 자료가 이 노드에 제공한 근거 유형(SUPPORTS/REFUTES일 때만).",
    )
    strength: EvidenceStrength
    rationale_dropped: int = Field(
        default=0, description="원문 대조에서 비검증(환각 의심)으로 걸러낸 문장 수."
    )
    rationale_repaired: int = Field(
        default=0,
        description="표기 차이(따옴표·띄어쓰기 등)를 보정해 원문 구간으로 바꿔 넣은 문장 수.",
    )
    rationale_offsets: list[list[int]] = Field(
        default_factory=list, description="근거 문장의 원문 위치 [start, end) (하이라이트용)."
    )
    downgraded: bool = Field(
        default=False,
        description="모델은 지지/반증으로 봤지만 원문에서 근거를 확인하지 못해 약한 관련으로 낮췄는지.",
    )
    candidate_via: str = Field(
        default="embedding", description="후보로 뽑힌 경로: embedding | keyword | both"
    )


class FailedNode(BaseModel):
    """판정하지 못한 노드. '무관'과 섞이지 않도록 따로 둔다."""

    node_id: str
    hypothesis: str
    code: str
    message: str


class ContentAnalysis(BaseModel):
    project_id: str
    source_url: str | None = None
    chunk_count: int
    candidate_node_ids: list[str]
    matched_nodes: list[MatchedNode]
    failed_nodes: list[FailedNode] = Field(default_factory=list)
    # 판정 출처 기록: 같은 자료의 결과가 왜 달라졌는지 나중에 추적할 수 있게 한다.
    model: str | None = None
    prompt_version: str | None = None
    analyzed_at: str | None = None
    content_chars: int = 0
    input_mode: str = Field(
        default="full", description="full(노드마다 전문) | excerpt(관련 부분 발췌)"
    )
