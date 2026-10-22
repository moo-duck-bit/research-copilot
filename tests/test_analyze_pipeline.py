"""analyze_content 파이프라인을 실제 DB 위에서 끝까지 돌린다 (LLM·임베딩은 가짜).

카톡 분석에서 나온 운영 장애 시나리오를 재현하고, 고친 동작을 확인한다.
"""
import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.api.routes import get_embedder, get_llm
from app.core.db import SessionLocal
from app.judgment.analyze import start_analysis
from app.llm.base import BudgetExceededError, LLMUpstreamError
from app.main import app
from tests.fakes import FakeEmbedder, FakeLLM, create_project, drop_project, judgment, node

NODES = [
    node("1", "신규 출점이 매출 성장을 이끈다", desc="매장별 매출과 출점 수"),
    node("2", "온라인 채널 매출 비중이 커진다", desc="온라인 매출 비중"),
    node("3", "원가 상승으로 수익성이 나빠진다", desc="영업이익률 추이"),
]
DOC = (
    "A사는 작년에 매장 40곳을 새로 열었고 신규 출점 효과로 매출이 12% 늘었다. "
    "온라인 매출 비중은 18%에서 21%로 커졌다. 원재료 가격이 올라 영업이익률은 3.1%로 떨어졌다."
)
QUOTE = {
    "신규 출점이 매출 성장을 이끈다": "A사는 작년에 매장 40곳을 새로 열었고 신규 출점 효과로 매출이 12% 늘었다.",
    "온라인 채널 매출 비중이 커진다": "온라인 매출 비중은 18%에서 21%로 커졌다.",
    "원가 상승으로 수익성이 나빠진다": "원재료 가격이 올라 영업이익률은 3.1%로 떨어졌다.",
}


def supports_all(hyp, _prompt):
    return judgment("SUPPORTS", [QUOTE[hyp]]) if hyp in QUOTE else judgment("IRRELEVANT")


@pytest.fixture
def env():
    llm = FakeLLM(supports_all)
    emb = FakeEmbedder()
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_embedder] = lambda: emb
    with TestClient(app) as c:
        pid = c.portal.call(create_project, NODES)
        yield c, pid, llm, emb
        c.portal.call(drop_project, pid)
    app.dependency_overrides.clear()


def post(c, pid, content, **kw):
    return c.post("/api/analyze", json={"project_id": pid, "content": content, **kw})


# --- 입력 단계 ---
def test_malformed_project_id_is_404_not_500(env):
    c, _pid, llm, _ = env
    for path in ("/api/projects/not-a-uuid", "/api/projects/not-a-uuid/collected",
                 "/api/projects/not-a-uuid/contents", "/api/contents/not-a-uuid"):
        r = c.get(path)
        assert r.status_code == 404, path
    r = post(c, "not-a-uuid", DOC)
    assert r.status_code == 404 and r.json()["code"] == "invalid_project_id"
    r = c.get("/api/projects/00000000-0000-0000-0000-000000000000/collected")
    assert r.status_code == 404 and r.json()["code"] == "project_not_found"
    assert llm.prompts == []


@pytest.mark.parametrize(
    "content,status,code",
    [
        ("https://news.example.com/article/123", 422, "content_url_only"),
        ("   ​  ", 422, "content_empty"),
        ("매출 증가", 422, "content_too_short"),
        ("가" * 60_001, 413, "content_too_long"),
    ],
)
def test_bad_input_is_rejected_before_any_llm_call(env, content, status, code):
    c, pid, llm, emb = env
    r = post(c, pid, content)
    assert r.status_code == status and r.json()["code"] == code
    assert llm.prompts == [] and emb.calls == []


def test_nul_and_invisible_chars_no_longer_crash_on_save(env):
    c, pid, _llm, _ = env
    r = post(c, pid, DOC.replace("작년에", "작년\x00에").replace("온라인", "온​라인"))
    assert r.status_code == 200, r.text
    stored = c.portal.call(_stored_content, r.json()["content_id"])
    assert "\x00" not in stored and "​" not in stored
    sup = [m for m in r.json()["matched_nodes"] if m["stance"] == "SUPPORTS"]
    assert len(sup) == 3  # 정리된 원문과 근거 문장 대조도 통과


async def _stored_content(cid):
    async with SessionLocal() as s:
        return (await s.execute(text("SELECT content FROM content_items WHERE id=:i"), {"i": cid})).scalar()


# --- 중복 제출 ---
def test_same_document_twice_reuses_result_without_new_calls(env):
    c, pid, llm, _ = env
    first = post(c, pid, DOC).json()
    n = len(llm.prompts)
    second = post(c, pid, "  " + DOC.replace(" ", "  ") + "\n").json()  # 공백만 다름
    assert second["reused"] is True and second["content_id"] == first["content_id"]
    assert len(llm.prompts) == n
    rows = c.get(f"/api/projects/{pid}/contents").json()
    assert len(rows) == 1


# --- 부분 실패와 이어서 하기 ---
def test_one_node_failure_keeps_other_results_and_retry_redoes_only_it(env):
    c, pid, llm, _ = env
    overloaded = LLMUpstreamError("529", 529, code="llm_overloaded", retry_after=30)
    llm.decide = lambda hyp, p: overloaded if hyp.startswith("원가") else supports_all(hyp, p)

    r = post(c, pid, DOC)
    assert r.status_code == 200
    body = r.json()
    assert [f["node_id"] for f in body["failed_nodes"]] == ["3"]
    assert body["failed_nodes"][0]["code"] == "llm_overloaded"
    assert {m["node_id"] for m in body["matched_nodes"]} == {"1", "2"}

    llm.decide = supports_all
    before = len(llm.prompts)
    body = post(c, pid, DOC).json()
    assert body.get("reused") is not True and body["failed_nodes"] == []
    assert {m["node_id"] for m in body["matched_nodes"]} == {"1", "2", "3"}
    assert [h for h, _ in llm.prompts[before:]] == ["원가 상승으로 수익성이 나빠진다"]


def test_total_failure_is_recorded_and_can_be_retried(env):
    c, pid, llm, _ = env
    llm.decide = lambda h, p: LLMUpstreamError("529", 529, code="llm_overloaded", retry_after=30)
    r = post(c, pid, DOC)
    assert r.status_code == 503 and r.json()["code"] == "llm_overloaded"
    assert r.headers["retry-after"] == "30"
    row = c.get(f"/api/projects/{pid}/contents").json()[0]
    assert row["status"] == "failed" and row["error"]["code"] == "llm_overloaded"

    llm.decide = supports_all
    r = post(c, pid, DOC)
    assert r.status_code == 200 and len(r.json()["matched_nodes"]) == 3


def test_budget_precheck_stops_before_spending(env):
    c, pid, llm, _ = env
    llm.budget_error = BudgetExceededError(1_990_000, 2_000_000, needed=20_000)
    r = post(c, pid, DOC)
    assert r.status_code == 429 and r.json()["code"] == "budget_exceeded"
    assert r.json()["needed"] == 20_000
    assert llm.prompts == []


# --- 비동기 접수와 새로고침 ---
def test_job_endpoint_returns_immediately_and_result_is_pollable(env):
    c, pid, _llm, _ = env
    r = c.post("/api/analyze/jobs", json={"project_id": pid, "content": DOC})
    assert r.status_code == 202 and r.json()["status"] == "processing"
    cid = r.json()["content_id"]
    deadline = time.time() + 10
    while time.time() < deadline:
        view = c.get(f"/api/contents/{cid}").json()
        if view["status"] != "processing":
            break
        time.sleep(0.1)
    assert view["status"] == "done" and len(view["result"]["matched_nodes"]) == 3
    # 새로고침 후 목록에서 그대로 다시 읽힌다
    assert c.get(f"/api/projects/{pid}/contents").json()[0]["content_id"] == cid
    # 같은 자료를 다시 접수하면 바로 저장된 결과
    again = c.post("/api/analyze/jobs", json={"project_id": pid, "content": DOC})
    assert again.status_code == 200 and again.json()["reused"] is True


def test_in_flight_rows_are_hidden_from_collected_and_block_duplicates(env):
    c, pid, _llm, _ = env
    job = c.portal.call(lambda: start_analysis(pid, DOC))
    assert job.kind == "run"
    view = c.get(f"/api/projects/{pid}/collected").json()
    assert view["summary"]["total_content_items"] == 0  # 처리 중인 빈 결과를 미분류로 세지 않는다
    r = post(c, pid, DOC)
    assert r.status_code == 409 and r.json()["code"] == "analysis_in_progress"
    assert r.json()["content_id"] == job.content_id


# --- 긴 자료 발췌, 키워드 보완, 출처 기록 ---
def test_long_document_sends_excerpts_but_verifies_against_full_text(env):
    c, pid, llm, _ = env
    filler = " ".join(f"{i}번째 일반 서술 문장이다." for i in range(900))
    long_doc = filler[:6000] + " " + QUOTE["온라인 채널 매출 비중이 커진다"] + " " + filler[6000:]
    assert len(long_doc) > 8_000
    body = post(c, pid, long_doc).json()
    assert body["input_mode"] == "excerpt"
    prompt = next(p for h, p in llm.prompts if h.startswith("온라인"))
    assert QUOTE["온라인 채널 매출 비중이 커진다"] in prompt and "[… 중략 …]" in prompt
    assert len(prompt) < len(long_doc)
    m = next(m for m in body["matched_nodes"] if m["node_id"] == "2")
    assert m["stance"] == "SUPPORTS" and m["rationale_sentences"]
    a, b = m["rationale_offsets"][0]
    assert long_doc[a:b] == m["rationale_sentences"][0]


def test_unverifiable_quote_is_downgraded_and_flagged(env):
    c, pid, llm, _ = env
    llm.decide = lambda h, p: judgment("SUPPORTS", ["원문에 없는 그럴듯한 문장을 지어냈다."])
    body = post(c, pid, DOC).json()
    for m in body["matched_nodes"]:
        assert m["stance"] == "WEAK_RELATED" and m["downgraded"] is True
        assert m["rationale_dropped"] == 1
    assert body["model"] == "fake-model" and body["prompt_version"].startswith("stance-")


def test_keyword_path_adds_node_that_embeddings_rank_out(env):
    c, pid, llm, _ = env
    extra = [node(str(i), f"무관한 가설 {i}번", desc=f"기타 지표 {i}") for i in range(4, 9)]
    extra.append(node("9", "EBITDA 마진이 개선된다", desc="EBITDA 마진 추이"))
    pid2 = c.portal.call(create_project, NODES + extra)
    try:
        llm.decide = lambda h, p: judgment("IRRELEVANT")
        c.app.dependency_overrides[get_embedder] = lambda: _FlatEmbedder()
        body = post(c, pid2, "올해 EBITDA 마진은 2024년 18%로 개선됐다고 회사가 밝혔다.").json()
        assert "9" in body["candidate_node_ids"]
        assert next(m for m in body["matched_nodes"] if m["node_id"] == "9")["candidate_via"] == "keyword"
    finally:
        c.portal.call(drop_project, pid2)


class _FlatEmbedder(FakeEmbedder):
    """노드 1~5만 질의와 가깝게 만드는 임베딩: 9번 노드는 임베딩으로는 절대 후보에 못 든다."""

    def _vec(self, t):
        v = [0.0] * self.dim
        if t.startswith("무관한") or "출점" in t or "온라인" in t or "원가" in t:
            v[0] = 1.0
        elif "EBITDA" in t and "마진이 개선된다" in t:
            v[1] = 1.0
        else:
            v[0] = 1.0  # 자료 청크(질의)
        return v


def test_short_document_fills_candidate_slots(env):
    # 청크가 1개인 짧은 자료도 청크당 3개에 묶이지 않고 후보 상한(5)까지 본다.
    c, _pid, llm, _ = env
    many = NODES + [node(str(i), f"추가 가설 {i}번 매출", desc=f"매출 지표 {i}") for i in range(4, 9)]
    pid2 = c.portal.call(create_project, many)
    try:
        llm.decide = lambda h, p: judgment("IRRELEVANT")
        body = post(c, pid2, DOC).json()
        assert body["chunk_count"] == 1 and len(body["candidate_node_ids"]) == 5
    finally:
        c.portal.call(drop_project, pid2)
