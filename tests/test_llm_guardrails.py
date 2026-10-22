"""LLM·임베딩 호출 보호 추가분 (실제 호출 없음).

샘플링 파라미터 미전송·thinking 끔, 출력 잘림 감지, 동시 호출 상한, 4xx 원인 분류, 사전 예산 확인,
tool 입력의 이중 직렬화 보정, 강도 등급, 임베딩 배치·대기 상한.
"""
import asyncio
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from app.core.config import settings
from app.llm.base import (
    BudgetExceededError,
    EmbeddingUnavailableError,
    LLMOutputTruncatedError,
    LLMUpstreamError,
)
from app.llm.claude import ClaudeProvider
from app.llm.embeddings import VoyageEmbeddingProvider, batches
from app.llm.resilience import CircuitBreaker
from app.llm.usage import InMemoryUsageLedger
from app.judgment.strength import score_strength
from app.schemas.analysis import Stance, StanceJudgment, StrengthFactors
from app.schemas.hypothesis_tree import HypothesisTree

REQ = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def resp(stop="tool_use"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", name="t", input={"x": 1})],
        stop_reason=stop,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


class Client:
    def __init__(self, outcomes=None, delay=0.0):
        self.outcomes = list(outcomes or [])
        self.kwargs = []
        self.delay = delay
        self.inflight = 0
        self.max_inflight = 0
        self.messages = self

    async def create(self, **kw):
        self.kwargs.append(kw)
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            o = self.outcomes.pop(0) if self.outcomes else resp()
            if isinstance(o, BaseException):
                raise o
            return o
        finally:
            self.inflight -= 1


def provider(client, used=0, budget=0, breaker=None):
    async def no_sleep(_):
        return None

    return ClaudeProvider(
        api_key="test", client=client, ledger=InMemoryUsageLedger(used),
        daily_token_budget=budget, sleep=no_sleep,
        breaker=breaker or CircuitBreaker(threshold=5, cooldown=60),
    )


def status_err(code, msg):
    cls = {400: anthropic.BadRequestError, 401: anthropic.AuthenticationError,
           404: anthropic.NotFoundError}.get(code, anthropic.APIStatusError)
    return cls(msg, response=httpx.Response(code, request=REQ), body=None)


def test_no_sampling_params_and_thinking_disabled():
    # claude-sonnet-5는 기본값이 아닌 temperature/top_p/top_k를 400으로 거부하고,
    # thinking을 생략하면 adaptive thinking이 켜진다.
    c = Client()
    asyncio.run(provider(c).extract(system="s", user="u", tools=[]))
    kw = c.kwargs[0]
    assert not {"temperature", "top_p", "top_k"} & kw.keys()
    assert kw["thinking"] == {"type": "disabled"}


def test_truncated_output_raises_instead_of_semantic_retry():
    b = CircuitBreaker(threshold=1, cooldown=60)
    c = Client([resp(stop="max_tokens")])
    with pytest.raises(LLMOutputTruncatedError):
        asyncio.run(provider(c, breaker=b).extract(system="s", user="u", tools=[]))
    assert len(c.kwargs) == 1 and b.state == "closed"  # 장애로 세지 않는다


def test_concurrent_calls_are_capped(monkeypatch):
    monkeypatch.setattr(settings, "llm_max_concurrency", 2)
    c = Client(delay=0.05)
    p = provider(c)

    async def burst():
        await asyncio.gather(*(p.extract(system="s", user="u", tools=[]) for _ in range(6)))

    asyncio.run(burst())
    assert len(c.kwargs) == 6 and c.max_inflight == 2


@pytest.mark.parametrize(
    "code,msg,expected",
    [
        (401, "invalid x-api-key", "llm_auth"),
        (400, "Your credit balance is too low to access the API", "llm_billing"),
        (400, "prompt is too long: 250000 tokens > 200000 maximum", "llm_input_too_long"),
        (404, "model: claude-x not found", "llm_model_unavailable"),
        (400, "tools.0.input_schema: invalid", "llm_bad_request"),
    ],
)
def test_fatal_errors_are_classified_by_cause(code, msg, expected):
    c = Client([status_err(code, msg)])
    with pytest.raises(LLMUpstreamError) as ei:
        asyncio.run(provider(c).extract(system="s", user="u", tools=[]))
    assert ei.value.code == expected and len(c.kwargs) == 1


def test_overload_after_retries_reports_overloaded_code():
    e = anthropic.OverloadedError("overloaded", response=httpx.Response(529, request=REQ), body=None)
    c = Client([e, e, e])
    with pytest.raises(LLMUpstreamError) as ei:
        asyncio.run(provider(c).extract(system="s", user="u", tools=[]))
    assert ei.value.code == "llm_overloaded" and ei.value.retry_after


def test_budget_precheck_blocks_before_any_call():
    c = Client()
    p = provider(c, used=900, budget=1_000)
    with pytest.raises(BudgetExceededError) as ei:
        asyncio.run(p.ensure_budget(200))
    assert ei.value.needed == 200 and ei.value.resets_at is not None
    assert c.kwargs == []
    asyncio.run(p.ensure_budget(50))  # 남은 100으로 충분하면 통과


# --- tool 입력 이중 직렬화 ---
def test_stance_fields_sent_as_json_strings_are_unwrapped():
    j = StanceJudgment.model_validate(
        {
            "stance": "SUPPORTS",
            "rationale_sentences": '["매출이 12% 늘었다."]',
            "warrant": "w",
            "provided_evidence_types": '["financial_performance"]',
            "factors": '{"evidence_type_fit": 0.8, "recency": null}',
        }
    )
    assert j.rationale_sentences == ["매출이 12% 늘었다."]
    assert j.factors.evidence_type_fit == 0.8


def test_tree_sent_as_json_string_is_unwrapped():
    import json

    node = {
        "node_id": "1", "parent_id": None, "level": 1, "hypothesis": "h",
        "sub_questions": ["q"], "priority": 3, "is_leaf": True, "mece_rationale": "m",
        "required_evidence": json.dumps(
            [{"evidence_type": "survey", "specificity": "qualitative", "recency_required_months": 0}]
        ),
    }
    node2 = {**node, "node_id": "2"}
    t = HypothesisTree.model_validate(
        {"primary_question": "p", "hypothesis_tree": json.dumps([node, node2], ensure_ascii=False)}
    )
    assert len(t.hypothesis_tree) == 2 and t.hypothesis_tree[0].required_evidence[0].evidence_type.value == "survey"


# --- 강도 등급 ---
def test_single_factor_cannot_reach_strong_band():
    one = score_strength(StrengthFactors(evidence_type_fit=0.95), Stance.SUPPORTS)
    assert one.score == 0.95 and one.band == "중" and one.assessed_count == 1
    two = score_strength(StrengthFactors(evidence_type_fit=0.9, scale=0.8), Stance.SUPPORTS)
    assert two.band == "강"
    none = score_strength(StrengthFactors(), Stance.REFUTES)
    assert none.score == 0.0 and none.band == "없음"


# --- 임베딩 ---
def test_embedding_batches_respect_item_and_char_limits():
    texts = ["가" * 300] * 10
    out = batches(texts, max_items=4, max_chars=1000)
    assert [len(b) for b in out] == [3, 3, 3, 1]
    assert sum(out, []) == texts


class RateLimited(Exception):
    http_status = 429


def test_embedding_rate_limit_wait_is_capped(monkeypatch):
    monkeypatch.setattr(settings, "embedding_rate_limit_wait_seconds", 20.0)
    monkeypatch.setattr(settings, "embedding_max_wait_seconds", 45.0)
    slept = []

    async def fake_sleep(s):
        slept.append(s)

    class C:
        calls = 0

        async def embed(self, texts, model, input_type):
            C.calls += 1
            raise RateLimited()

    p = VoyageEmbeddingProvider(api_key="k", client=C(), sleep=fake_sleep)
    with pytest.raises(EmbeddingUnavailableError):
        asyncio.run(p.embed(["a"]))
    assert sum(slept) <= 45.0 and C.calls == 3
