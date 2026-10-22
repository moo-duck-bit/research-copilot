"""LLM 호출 보호: 오류 코드별 재시도, 서킷브레이커, 일일 토큰 예산 (실제 호출 없음)."""
import asyncio
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from app.llm.base import BudgetExceededError, CircuitOpenError, LLMUpstreamError
from app.llm.claude import ClaudeProvider, _shared_client
from app.llm.resilience import CircuitBreaker
from app.llm.usage import InMemoryUsageLedger

REQ = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
CLASSES = {
    400: anthropic.BadRequestError,
    429: anthropic.RateLimitError,
    500: anthropic.InternalServerError,
    529: anthropic.OverloadedError,
}


def err(code, retry_after=None):
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    resp = httpx.Response(code, request=REQ, headers=headers)
    return CLASSES.get(code, anthropic.APIStatusError)(f"status {code}", response=resp, body=None)


def ok():
    return SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", name="t", input={"x": 1})],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
    )


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.messages = self

    async def create(self, **kwargs):
        self.calls += 1
        o = self.outcomes.pop(0)
        if isinstance(o, BaseException):
            raise o
        return o


def make(outcomes, *, breaker=None, used=0, budget=0):
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    client = FakeClient(outcomes)
    ledger = InMemoryUsageLedger(used)
    p = ClaudeProvider(
        api_key="test",
        client=client,
        breaker=breaker or CircuitBreaker(threshold=5, cooldown=60),
        ledger=ledger,
        daily_token_budget=budget,
        sleep=fake_sleep,
    )
    return p, client, sleeps, ledger


def call(p):
    return asyncio.run(p.extract(system="s", user="u", tools=[]))


def test_sdk_auto_retry_is_disabled():
    assert _shared_client("k").max_retries == 0


def test_bad_request_is_not_retried():
    p, client, sleeps, _ = make([err(400)])
    with pytest.raises(LLMUpstreamError):
        call(p)
    assert client.calls == 1 and sleeps == []


def test_rate_limit_waits_retry_after_then_succeeds():
    p, client, sleeps, _ = make([err(429, retry_after=2), ok()])
    assert call(p) == ("t", {"x": 1})
    assert client.calls == 2 and sleeps == [2.0]


def test_rate_limit_with_long_retry_after_fails_fast():
    p, client, sleeps, _ = make([err(429, retry_after=120)])
    with pytest.raises(LLMUpstreamError):
        call(p)
    assert client.calls == 1 and sleeps == []


def test_rate_limit_does_not_trip_breaker():
    b = CircuitBreaker(threshold=2, cooldown=60)
    p, client, _, _ = make([err(429, 0), err(429, 0), err(429, 0)], breaker=b)
    with pytest.raises(LLMUpstreamError):
        call(p)
    assert client.calls == 3 and b.state == "closed"


def test_overloaded_retries_with_backoff_then_gives_up():
    p, client, sleeps, _ = make([err(529), err(529), err(529)])
    with pytest.raises(LLMUpstreamError):
        call(p)
    assert client.calls == 3 and len(sleeps) == 2
    assert sleeps[1] > sleeps[0] - 1.0  # 지수 백오프(지터 포함)


def test_connection_error_is_retried():
    p, client, _, _ = make([anthropic.APIConnectionError(request=REQ), ok()])
    assert call(p)[0] == "t"
    assert client.calls == 2


def test_breaker_opens_then_fails_fast_then_recovers():
    now = [0.0]
    b = CircuitBreaker(threshold=3, cooldown=60, clock=lambda: now[0])
    p, client, _, _ = make([err(529), err(529), err(529), ok()], breaker=b)

    with pytest.raises(LLMUpstreamError):
        call(p)  # 3번 연속 529 → 열림
    assert b.state == "open" and client.calls == 3

    with pytest.raises(CircuitOpenError):
        call(p)  # 열려 있는 동안은 호출 자체를 보내지 않는다
    assert client.calls == 3

    now[0] = 61.0  # cooldown 경과 → 시험 호출 1번 허용
    assert call(p)[0] == "t"
    assert b.state == "closed" and client.calls == 4


def test_budget_exceeded_blocks_call():
    p, client, _, _ = make([ok()], used=1_000, budget=1_000)
    with pytest.raises(BudgetExceededError):
        call(p)
    assert client.calls == 0


def test_successful_call_records_usage():
    p, _, _, ledger = make([ok()], used=0, budget=1_000)
    call(p)
    assert ledger.added == [(100, 20)]
