"""LLM 호출 추상화. 판단 로직은 이 얇은 Protocol 뒤에서만 LLM에 접근한다.

구현체는 서버 내부에만 두며, 클라이언트로 노출하지 않는다.

오류마다 code를 붙인다. 사용자 화면과 MCP 클라이언트가 '공급사 장애', '예산 소진',
'입력 문제', '서버 설정 문제'를 구분해 안내할 수 있게 하기 위해서다.
"""
import datetime
from typing import Any, Protocol, runtime_checkable


class LLMError(RuntimeError):
    """LLM 호출 일반 오류."""

    code = "llm_error"


class LLMConfigError(LLMError):
    """설정 오류(예: API 키 미설정)."""

    code = "llm_config"


class LLMUpstreamError(LLMError):
    """재시도 정책을 다 쓰고도 상위 API 호출이 실패했을 때.

    code 예: llm_overloaded(529·5xx 지속), llm_rate_limited(429 지속), llm_auth(401/403, 키·권한),
    llm_billing(크레딧·결제), llm_input_too_long(입력 길이 초과), llm_model_unavailable(모델 없음·종료),
    llm_bad_request(그 밖의 400).
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        *,
        code: str = "llm_upstream",
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.retry_after = retry_after


class LLMOutputTruncatedError(LLMError):
    """출력이 max_tokens에서 잘렸다. 같은 설정으로 다시 보내도 같은 곳에서 잘린다."""

    code = "llm_output_truncated"


class CircuitOpenError(LLMError):
    """서킷브레이커가 열려 있어 호출을 보내지 않았다."""

    code = "llm_circuit_open"

    def __init__(self, retry_after: float) -> None:
        super().__init__(f"LLM 서킷브레이커가 열려 있다. {retry_after:.0f}초 뒤 다시 시도하라.")
        self.retry_after = retry_after


class BudgetExceededError(LLMError):
    """오늘 토큰 예산이 모자라 호출을 보내지 않았다."""

    code = "budget_exceeded"

    def __init__(
        self,
        used: int,
        budget: int,
        *,
        needed: int | None = None,
        resets_at: datetime.datetime | None = None,
    ) -> None:
        if needed:
            msg = (
                f"이 자료를 분석하려면 약 {needed:,} 토큰이 필요한데 오늘 남은 예산은 "
                f"{max(0, budget - used):,} 토큰이다."
            )
        else:
            msg = f"오늘 LLM 토큰 예산을 다 썼다 (사용 {used:,} / 예산 {budget:,})."
        super().__init__(msg)
        self.used = used
        self.budget = budget
        self.needed = needed
        self.resets_at = resets_at


class EmbeddingUnavailableError(LLMError):
    """임베딩 API가 한도·장애로 제한 시간 안에 응답하지 못했다."""

    code = "embedding_unavailable"

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


@runtime_checkable
class LLMProvider(Protocol):
    """서버 내부 판단 로직이 사용하는 최소 LLM 인터페이스."""

    async def extract(
        self,
        *,
        system: str,
        user: str,
        tools: list[dict[str, Any]],
        tool_choice: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """tool-use(function calling)로 구조화 출력을 강제한다.

        모델이 호출한 (tool_name, tool_input) 튜플을 반환한다.
        tool_input은 JSON 스키마를 따르지만 Pydantic 검증은 호출자가 수행한다.
        """
        ...
