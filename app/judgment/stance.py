"""3단계: 스탠스 분류 + rationale 추출 + warrant 생성 (Claude tool-use + Pydantic 검증)."""
from __future__ import annotations

from pydantic import ValidationError

from app.llm.base import LLMError, LLMProvider
from app.prompts.analyze import STANCE_SYSTEM, build_stance_user
from app.schemas.analysis import StanceJudgment

_TOOL_NAME = "classify_stance"


class StanceClassificationError(LLMError):
    """스탠스 판정이 재시도 후에도 스키마 검증에 실패했을 때."""

    code = "llm_schema_invalid"


def _tool() -> dict:
    return {
        "name": _TOOL_NAME,
        "description": "자료가 가설의 참/거짓에 주는 증거를 스탠스로 판정한다.",
        "input_schema": StanceJudgment.model_json_schema(),
    }


async def classify_stance(
    document: str, node: dict, provider: LLMProvider, *, excerpt: bool = False
) -> StanceJudgment:
    """(자료, 노드) 한 쌍을 판정한다. 검증 실패 시 1회 재시도.

    주: 추론(thinking) 모드는 켜지 않는다 — 근거 추출에서 환각을 늘리므로.
    (ClaudeProvider.extract가 thinking을 명시적으로 끈다. 생략하면 claude-sonnet-5는 켜진다.)
    """
    tools = [_tool()]
    tool_choice = {"type": "tool", "name": _TOOL_NAME}
    correction: str | None = None
    last_error: Exception | None = None

    for _attempt in range(2):
        user = build_stance_user(document, node, correction, excerpt=excerpt)
        name, data = await provider.extract(
            system=STANCE_SYSTEM, user=user, tools=tools, tool_choice=tool_choice
        )
        try:
            if name != _TOOL_NAME:
                raise ValueError(f"예상치 못한 도구 호출: {name}")
            return StanceJudgment.model_validate(data)
        except (ValidationError, ValueError) as exc:
            last_error = exc
            correction = str(exc)

    raise StanceClassificationError(
        f"스탠스 판정 스키마 검증이 재시도 후에도 실패했다: {last_error}"
    )
