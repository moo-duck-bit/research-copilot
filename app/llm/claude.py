"""Anthropic Claude 구현체 (프로토타입: Sonnet 5 단일 모델).

tool-use(function calling)로 JSON 스키마를 강제한다. Pydantic 검증/재시도는
판단 로직(app/judgment) 쪽에서 수행한다. API 키는 환경변수에서만 받는다.

호출 보호(비용 급증 방지):
- SDK 자동 재시도는 끈다(max_retries=0). 429와 529를 구분하지 않고 재시도하기 때문이다.
- 오류 코드별 재시도는 app/llm/resilience.py의 분류를 따른다.
- 연속 장애가 쌓이면 서킷브레이커가 호출을 막는다(프로세스당 하나).
- 호출 전 일일 토큰 예산을 확인하고, 성공하면 사용량을 DB 원장에 더한다.
- 프로세스 전체의 동시 호출 수에 상한을 둔다(자료 1건이 노드 수만큼 병렬로 부르므로,
  사용자가 몇 명만 겹쳐도 429가 난다).
- 샘플링 파라미터(temperature 등)는 보내지 않는다. claude-sonnet-5는 기본값이 아닌 값을 400으로
  거부한다. 같은 자료에 같은 판정을 주는 장치는 저장된 결과 재사용(app/judgment/analyze.py)이다.
- thinking은 명시적으로 끈다. claude-sonnet-5는 thinking을 생략하면 adaptive thinking이 켜져
  근거 추출 지시와 어긋나고, 생각 토큰이 max_tokens를 먹는다. (claude-sonnet-5-5로 바꿀 때는
  'disabled'가 400이므로 {"type": "between_tools"}로 바꿔야 한다.)
- 출력이 max_tokens에서 잘리면(stop_reason=max_tokens) 검증 재시도로 넘기지 않고 바로 알린다.
  같은 설정으로 다시 보내면 같은 곳에서 또 잘리기 때문이다.
"""
from __future__ import annotations

import asyncio
import weakref
from collections.abc import Awaitable, Callable
from typing import Any

from anthropic import AsyncAnthropic

from app.core.config import settings
from app.llm.base import (
    BudgetExceededError,
    LLMConfigError,
    LLMError,
    LLMOutputTruncatedError,
    LLMUpstreamError,
)
from app.llm.resilience import (
    FATAL,
    RATE_LIMIT,
    CircuitBreaker,
    backoff_seconds,
    classify,
    retry_after_seconds,
)
from app.llm.usage import DbUsageLedger, UsageLedger, next_reset, precheck_budget

# tool-use로 출력을 강제할 때의 최대 토큰. 큰 트리를 담기 위해 넉넉히 잡는다.
_MAX_TOKENS = 8192

# 프로세스(파드)마다 하나씩 공유한다. 요청마다 새로 만들면 브레이커가 장애를 기억하지 못하고,
# 클라이언트를 매번 만들면 커넥션 풀이 계속 늘어난다.
_breaker = CircuitBreaker(
    threshold=settings.llm_breaker_failure_threshold,
    cooldown=settings.llm_breaker_cooldown_seconds,
)
_clients: dict[str, AsyncAnthropic] = {}


def default_breaker() -> CircuitBreaker:
    return _breaker


# 동시 호출 상한. 세마포어는 이벤트 루프에 묶이므로 루프마다 하나씩 둔다.
_semaphores: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = (
    weakref.WeakKeyDictionary()
)


def _concurrency_gate() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    sem = _semaphores.get(loop)
    if sem is None:
        sem = asyncio.Semaphore(max(1, settings.llm_max_concurrency))
        _semaphores[loop] = sem
    return sem


def fatal_code(status: int | None, message: str) -> str:
    """재시도하지 않는 4xx를 원인별로 나눈다. 사용자 안내와 운영 알림이 달라야 해서다."""
    m = message.lower()
    if status in (401, 403):
        return "llm_auth"
    if status == 404 or "model" in m and ("not found" in m or "deprecated" in m):
        return "llm_model_unavailable"
    if "credit" in m or "billing" in m or "balance" in m:
        return "llm_billing"
    if status == 413 or "too long" in m or "too many tokens" in m or "context" in m and "exceed" in m:
        return "llm_input_too_long"
    return "llm_bad_request"


def _shared_client(api_key: str) -> AsyncAnthropic:
    if api_key not in _clients:
        _clients[api_key] = AsyncAnthropic(
            api_key=api_key, max_retries=0, timeout=settings.llm_timeout_seconds
        )
    return _clients[api_key]


class ClaudeProvider:
    """LLMProvider Protocol 구현체. 유일한 구현체이며 서버 내부에만 존재한다."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        *,
        client: Any | None = None,
        breaker: CircuitBreaker | None = None,
        ledger: UsageLedger | None = None,
        daily_token_budget: int | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        # 키는 환경변수(ANTHROPIC_API_KEY) 경유. 하드코딩 금지.
        self._api_key = api_key or settings.anthropic_api_key
        self._model = model or settings.claude_model
        self._client = client
        self._breaker = breaker or _breaker
        self._ledger = ledger or DbUsageLedger(settings.budget_utc_offset_hours)
        self._budget = (
            settings.daily_token_budget if daily_token_budget is None else daily_token_budget
        )
        self._sleep = sleep

    @property
    def model(self) -> str:
        return self._model

    async def ensure_budget(self, needed_tokens: int) -> None:
        """여러 번 호출할 작업을 시작하기 전에, 끝까지 할 예산이 남았는지 본다."""
        await precheck_budget(
            self._ledger, self._budget, needed_tokens, settings.budget_utc_offset_hours
        )

    async def _check_budget(self) -> None:
        if self._budget <= 0:
            return
        used = await self._ledger.used_today()
        if used >= self._budget:
            raise BudgetExceededError(
                used, self._budget, resets_at=next_reset(settings.budget_utc_offset_hours)
            )

    async def extract(
        self,
        *,
        system: str,
        user: str,
        tools: list[dict[str, Any]],
        tool_choice: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        if self._client is None and not self._api_key:
            raise LLMConfigError("ANTHROPIC_API_KEY가 설정되지 않았다. 환경변수로 주입하라.")
        client = self._client or _shared_client(self._api_key)
        # 단일 도구 호출만 받도록 병렬 tool-use를 끈다.
        choice = tool_choice or {"type": "any", "disable_parallel_tool_use": True}

        await self._check_budget()

        rate_limit_tries = 0
        transient_tries = 0
        while True:
            self._breaker.before_call()  # 열려 있으면 CircuitOpenError, 호출 안 보냄
            try:
                async with _concurrency_gate():
                    resp = await client.messages.create(
                        model=self._model,
                        max_tokens=_MAX_TOKENS,
                        thinking={"type": "disabled"},
                        system=system,
                        messages=[{"role": "user", "content": user}],
                        tools=tools,
                        tool_choice=choice,
                    )
            except Exception as exc:  # noqa: BLE001 - 분류 후 다시 던진다
                kind = classify(exc)
                status = getattr(exc, "status_code", None)

                if kind == FATAL:
                    self._breaker.record_neutral()
                    raise LLMUpstreamError(
                        f"재시도하지 않는 오류(status={status}): {exc}",
                        status,
                        code=fatal_code(status, str(exc)),
                    ) from exc

                if kind == RATE_LIMIT:
                    self._breaker.record_neutral()
                    wait = retry_after_seconds(exc)
                    if wait is None:
                        wait = backoff_seconds(rate_limit_tries + 1, settings.llm_backoff_base_seconds)
                    if (
                        rate_limit_tries >= settings.llm_rate_limit_retries
                        or wait > settings.llm_retry_after_cap_seconds
                    ):
                        raise LLMUpstreamError(
                            f"요청 한도 초과(429), 대기 {wait:.0f}초 필요: 재시도 중단",
                            429,
                            code="llm_rate_limited",
                            retry_after=wait,
                        ) from exc
                    rate_limit_tries += 1
                    await self._sleep(wait)
                    continue

                # TRANSIENT: 529·5xx·408·연결 오류
                self._breaker.record_failure()
                if transient_tries >= settings.llm_transient_retries:
                    raise LLMUpstreamError(
                        f"일시 장애가 이어져 재시도 중단(status={status}): {exc}",
                        status,
                        code="llm_overloaded",
                        retry_after=30.0,
                    ) from exc
                transient_tries += 1
                await self._sleep(
                    backoff_seconds(transient_tries, settings.llm_backoff_base_seconds)
                )
                continue

            self._breaker.record_success()
            usage = getattr(resp, "usage", None)
            if usage is not None:
                await self._ledger.add(
                    int(getattr(usage, "input_tokens", 0) or 0),
                    int(getattr(usage, "output_tokens", 0) or 0),
                )
            if getattr(resp, "stop_reason", None) == "max_tokens":
                raise LLMOutputTruncatedError(
                    f"출력이 최대 길이({_MAX_TOKENS} 토큰)에서 잘렸다. 입력을 줄여 다시 시도하라."
                )
            for block in resp.content:
                if block.type == "tool_use":
                    return block.name, dict(block.input)
            raise LLMError(
                f"모델이 tool_use를 반환하지 않았다 (stop_reason={resp.stop_reason})."
            )
