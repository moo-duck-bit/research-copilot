"""4단계: 근거 강도 스코어링.

각 요인(0~1)의 가중합으로 단일 score를 계산하되, 자료에 정보가 없어 미평가(null)된
요인은 가중치 합에서 빼고 남은 요인만으로 재정규화한다. score는 모델이 임의로 뱉는 값이
아니라 factors로부터 서버가 결정론적으로 계산한다.
"""
from __future__ import annotations

from app.schemas.analysis import EvidenceStrength, Stance, StrengthFactors

# 증거유형 적합도를 가장 크게, 그다음 신뢰도/최신성/규모.
_WEIGHTS = {
    "evidence_type_fit": 0.40,
    "source_credibility": 0.25,
    "recency": 0.20,
    "scale": 0.15,
}


def strength_band(score: float, assessed_count: int) -> str:
    """점수를 강/중/약 등급으로 바꾼다.

    요인이 하나만 평가되면 그 값이 곧 점수가 되어 근거가 얇은데도 높게 나올 수 있다.
    그래서 평가된 요인이 2개 미만이면 등급을 '중'까지만 준다.
    """
    if score <= 0 or assessed_count == 0:
        return "없음"
    band = "강" if score >= 0.7 else "중" if score >= 0.4 else "약"
    if band == "강" and assessed_count < 2:
        band = "중"
    return band


def score_strength(factors: StrengthFactors, stance: Stance) -> EvidenceStrength:
    not_assessed = [k for k in _WEIGHTS if getattr(factors, k) is None]
    assessed = {k: getattr(factors, k) for k in _WEIGHTS if getattr(factors, k) is not None}

    # 실질 증거가 없는 스탠스는 강도 0. factors/not_assessed는 투명성 위해 보존.
    # 평가 가능한 요인이 하나도 없어도 0 (0으로 나누지 않는다).
    if stance in (Stance.WEAK_RELATED, Stance.IRRELEVANT) or not assessed:
        return EvidenceStrength(
            score=0.0, factors=factors, not_assessed=not_assessed,
            band="없음", assessed_count=len(assessed),
        )

    weight_sum = sum(_WEIGHTS[k] for k in assessed)  # 재정규화 분모
    score = round(sum(assessed[k] * _WEIGHTS[k] for k in assessed) / weight_sum, 3)
    return EvidenceStrength(
        score=score, factors=factors, not_assessed=not_assessed,
        band=strength_band(score, len(assessed)), assessed_count=len(assessed),
    )
