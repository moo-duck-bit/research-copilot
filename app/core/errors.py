"""사용자에게 돌려줄 오류의 모양을 한곳에서 정한다.

응답 본문은 {"detail": 사람이 읽을 문장, "code": 기계가 읽을 코드, ...추가 정보}.
웹 화면은 code로 안내 문구를 고르고, MCP 클라이언트는 detail을 그대로 보여 준다.
원인을 '공급사 장애 / 예산 / 입력 / 서버 설정'으로 나눠 알려야 사용자가 자기 탓인지,
기다리면 되는지, 관리자에게 알려야 하는지를 판단할 수 있다.
"""
from __future__ import annotations

from typing import Any

from app.llm.base import (
    BudgetExceededError,
    CircuitOpenError,
    EmbeddingUnavailableError,
    LLMConfigError,
    LLMError,
    LLMOutputTruncatedError,
    LLMUpstreamError,
)


class AppError(RuntimeError):
    def __init__(self, status: int, code: str, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.extra = extra


# LLM 오류 코드 → HTTP 상태
_UPSTREAM_STATUS = {
    "llm_overloaded": 503,
    "llm_rate_limited": 503,
    "llm_auth": 502,
    "llm_billing": 502,
    "llm_model_unavailable": 502,
    "llm_input_too_long": 413,
    "llm_bad_request": 502,
}


def describe(exc: BaseException) -> tuple[int, str, str, dict[str, Any]]:
    """예외 → (HTTP 상태, code, 메시지, 추가 정보)."""
    if isinstance(exc, AppError):
        return exc.status, exc.code, exc.message, exc.extra
    if isinstance(exc, BudgetExceededError):
        extra: dict[str, Any] = {"used": exc.used, "budget": exc.budget}
        if exc.needed:
            extra["needed"] = exc.needed
        if exc.resets_at:
            extra["resets_at"] = exc.resets_at.isoformat()
        return 429, exc.code, str(exc), extra
    if isinstance(exc, CircuitOpenError):
        return 503, exc.code, str(exc), {"retry_after": max(1, int(exc.retry_after))}
    if isinstance(exc, EmbeddingUnavailableError):
        extra = {"retry_after": int(exc.retry_after)} if exc.retry_after else {}
        return 503, exc.code, str(exc), extra
    if isinstance(exc, LLMUpstreamError):
        extra = {"retry_after": int(exc.retry_after)} if exc.retry_after else {}
        return _UPSTREAM_STATUS.get(exc.code, 502), exc.code, str(exc), extra
    if isinstance(exc, LLMOutputTruncatedError):
        return 502, exc.code, str(exc), {}
    if isinstance(exc, LLMConfigError):
        return 500, exc.code, str(exc), {}
    if isinstance(exc, LLMError):
        return 502, getattr(exc, "code", "llm_error"), str(exc), {}
    return 500, "internal_error", "서버 내부 오류가 발생했다.", {}


def error_code(exc: BaseException) -> str:
    return describe(exc)[1]


class ProjectNotFoundError(AppError):
    def __init__(self, message: str, code: str = "project_not_found") -> None:
        super().__init__(404, code, message)


def parse_project_id(project_id: str) -> str:
    """UUID 형식이 아니면 DB에 보내기 전에 404로 막는다.

    예전에는 'not-a-uuid'가 그대로 DB로 가서 asyncpg 형변환 오류가 500으로 새어 나갔다.
    MCP 클라이언트 모델이 UUID를 한 글자 틀리거나 지어내는 경우에도 원인을 알려 준다.
    """
    import uuid

    try:
        return str(uuid.UUID(str(project_id).strip()))
    except (ValueError, AttributeError):
        raise ProjectNotFoundError(
            f"project_id '{project_id}'는 올바른 형식이 아니다. "
            "set_project_brief가 돌려준 project_id를 그대로 써라.",
            code="invalid_project_id",
        ) from None
