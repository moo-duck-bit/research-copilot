"""analyze_content 파이프라인 오케스트레이터 (4단계 결합).

1) 후보 필터(임베딩·pgvector + 키워드 보완) → 2) rationale 추출 + 3) 스탠스/warrant(Claude)
→ 4) 근거 강도. 원문 대조 검증(환각 방지)까지 포함한다.

실행 흐름은 '접수'와 '실행' 두 단계로 나뉜다.
- 접수(start_analysis): 입력을 정리·검사하고, 같은 프로젝트에 같은 자료가 이미 있는지 본다.
  있으면 저장된 결과를 돌려주고(토큰 사용 없음), 없으면 '처리 중' 행을 먼저 만든다.
  그래서 새로고침하거나 창을 닫아도 작업이 목록에 남고, 같은 자료를 다시 눌러도 두 번 돌지 않는다.
- 실행(run_analysis): 노드별 판정. 한 노드가 실패해도 나머지 결과는 저장한다(부분 결과).
  다시 실행하면 실패한 노드만 다시 판정한다.
"""
from __future__ import annotations

import asyncio
import datetime
import logging
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from app.core.config import settings
from app.core.db import SessionLocal
from app.core.errors import AppError, ProjectNotFoundError, describe, parse_project_id
from app.core.text_safety import content_hash, is_url_only, sanitize_text
from app.judgment.stance import classify_stance
from app.judgment.strength import score_strength
from app.judgment.verbatim import verify_rationale
from app.llm.base import LLMProvider
from app.llm.embeddings import EmbeddingProvider
from app.llm.usage import estimate_tokens
from app.models.content import ContentItem
from app.prompts.analyze import PROMPT_VERSION
from app.retrieval.candidates import ensure_node_embeddings, retrieve_candidate_detail
from app.retrieval.chunking import OMIT_MARK, build_excerpt, chunk_spans
from app.schemas.analysis import ContentAnalysis, FailedNode, MatchedNode, Stance

log = logging.getLogger("app.analyze")

__all__ = ["ProjectNotFoundError", "analyze_content", "start_analysis", "run_analysis"]

# 노드 1회 판정의 프롬프트 고정부(지시문·가설·required_evidence)와 출력 어림치(토큰)
_PROMPT_OVERHEAD_TOKENS = 2_500
_OUTPUT_TOKENS = 800


class AnalysisInProgressError(AppError):
    def __init__(self, content_id: str) -> None:
        super().__init__(
            409,
            "analysis_in_progress",
            "같은 자료를 이미 분석하고 있다. 끝나면 결과가 목록에 나타난다.",
            content_id=content_id,
        )


def validate_content(raw: str) -> str:
    """정리한 자료 텍스트를 돌려준다. 분석할 수 없으면 LLM을 부르기 전에 거절한다."""
    content = sanitize_text(raw or "")
    if not content:
        raise AppError(422, "content_empty", "자료 본문이 비어 있다.")
    if is_url_only(content):
        raise AppError(
            422,
            "content_url_only",
            "URL만 들어왔다. 서버는 URL의 내용을 가져오지 않는다. 기사·리포트 본문을 붙여넣어라.",
        )
    if len(content) < settings.min_content_chars:
        raise AppError(
            422,
            "content_too_short",
            f"자료가 너무 짧다({len(content)}자). 판단할 수 있도록 {settings.min_content_chars}자"
            " 이상의 본문을 넣어라.",
            min_chars=settings.min_content_chars,
        )
    if len(content) > settings.max_content_chars:
        raise AppError(
            413,
            "content_too_long",
            f"자료가 너무 길다({len(content):,}자, 최대 {settings.max_content_chars:,}자)."
            " 관련 장(章)이나 부분만 나눠서 넣어라.",
            chars=len(content),
            max_chars=settings.max_content_chars,
        )
    return content


async def _load_project(project_id: str) -> dict:
    async with SessionLocal() as session:
        row = (
            await session.execute(
                text("SELECT tree FROM projects WHERE id=:pid"),
                {"pid": project_id},
            )
        ).first()
    if not row or not row[0]:
        raise ProjectNotFoundError(f"project_id={project_id} 트리를 찾을 수 없다.")
    return row[0]


@dataclass
class Job:
    kind: str  # reuse | busy | run
    content_id: str
    project_id: str
    content: str
    source_url: str | None
    result: dict | None = None  # reuse: 저장된 결과, run: 이어서 할 부분 결과(있으면)


async def start_analysis(
    project_id: str, raw_content: str, *, source_url: str | None = None, force: bool = False
) -> Job:
    """입력 검사 + 중복 확인 + '처리 중' 행 만들기. LLM은 부르지 않는다."""
    pid = parse_project_id(project_id)
    content = validate_content(raw_content)
    source_url = (sanitize_text(source_url) or None) if source_url else None
    await _load_project(pid)  # 없는 프로젝트면 여기서 404
    h = content_hash(content)

    async with SessionLocal() as s:
        async with s.begin():
            # 같은 자료가 동시에 두 번 들어와도 한쪽만 접수되도록 (프로젝트, 해시) 단위로 잠근다.
            await s.execute(
                text("SELECT pg_advisory_xact_lock(hashtext(:k))"), {"k": f"{pid}:{h}"}
            )
            row = (
                await s.execute(
                    text(
                        "SELECT id, status, result,"
                        " updated_at < now() - make_interval(secs => :stale) AS stale"
                        " FROM content_items WHERE project_id=:pid AND content_hash=:h"
                        " ORDER BY created_at DESC LIMIT 1"
                    ),
                    {"pid": pid, "h": h, "stale": settings.analysis_stale_after_seconds},
                )
            ).first()
            if row:
                cid, status, result, stale = str(row[0]), row[1], row[2] or {}, row[3]
                partial = bool(result.get("failed_nodes"))
                if status == "done" and not partial and not force:
                    return Job("reuse", cid, pid, content, source_url, result)
                if status == "processing" and not stale:
                    return Job("busy", cid, pid, content, source_url)
                # 실패했거나, 끊겼거나(오래된 처리 중), 부분 결과이거나, 강제 재분석 → 같은 행으로 다시
                await s.execute(
                    text(
                        "UPDATE content_items SET status='processing', error_code=NULL,"
                        " error_message=NULL, updated_at=now(),"
                        " source_url=COALESCE(:url, source_url) WHERE id=:id"
                    ),
                    {"id": cid, "url": source_url},
                )
                resume = result if (status == "done" and partial and not force) else None
                return Job("run", cid, pid, content, source_url, resume)

            item = ContentItem(
                id=uuid.uuid4(),
                project_id=uuid.UUID(pid),
                source_url=source_url,
                content=content,
                result={},
                content_hash=h,
                status="processing",
            )
            s.add(item)
            return Job("run", str(item.id), pid, content, source_url)


async def _mark_done(content_id: str, result: dict) -> None:
    async with SessionLocal() as s:
        item = await s.get(ContentItem, uuid.UUID(content_id))
        if item is None:
            return
        item.result = result
        item.status = "done"
        item.error_code = None
        item.error_message = None
        item.updated_at = datetime.datetime.now(datetime.timezone.utc)
        await s.commit()


async def _mark_failed(content_id: str, exc: BaseException) -> None:
    if isinstance(exc, asyncio.CancelledError):
        code, msg = "analysis_interrupted", "분석이 중간에 끊겼다(서버 재시작 등). 다시 시도하라."
    else:
        _status, code, msg, _ = describe(exc)
    try:
        async with SessionLocal() as s:
            await s.execute(
                text(
                    "UPDATE content_items SET status='failed', error_code=:c, error_message=:m,"
                    " updated_at=now() WHERE id=:id AND status='processing'"
                ),
                {"id": content_id, "c": code, "m": msg[:500]},
            )
            await s.commit()
    except Exception:  # noqa: BLE001 - 상태 기록 실패가 원래 오류를 가리지 않게
        log.exception("분석 실패 상태 기록 실패 content_id=%s", content_id)


async def run_analysis(
    job: Job, llm: LLMProvider, embedder: EmbeddingProvider, *, max_candidates: int = 5
) -> dict[str, Any]:
    try:
        result = await _pipeline(job, llm, embedder, max_candidates=max_candidates)
    except BaseException as exc:
        await _mark_failed(job.content_id, exc)
        raise
    await _mark_done(job.content_id, result)
    return {**result, "content_id": job.content_id, "status": "done"}


async def _pipeline(
    job: Job, llm: LLMProvider, embedder: EmbeddingProvider, *, max_candidates: int
) -> dict[str, Any]:
    content = job.content
    tree = await _load_project(job.project_id)
    nodes = tree["hypothesis_tree"]
    node_by_id = {n["node_id"]: n for n in nodes}

    # 1단계: 후보 필터
    await ensure_node_embeddings(job.project_id, nodes, embedder)
    chunks = chunk_spans(content)
    cands = await retrieve_candidate_detail(
        job.project_id, chunks, nodes, embedder, max_candidates=max_candidates
    )
    candidates = [(nid, sim) for nid, sim in cands.ranked if nid in node_by_id]

    # 긴 자료는 노드마다 전문 대신 관련 청크와 앞뒤만 보낸다(비용·지연·중간부 누락 방지).
    # 근거 문장 검증은 언제나 전문과 대조한다.
    excerpt_mode = len(content) > settings.full_document_max_chars

    def doc_for(nid: str) -> str:
        if not excerpt_mode:
            return content
        ex = build_excerpt(
            content, chunks, cands.chunks_by_node.get(nid, []),
            max_chars=settings.excerpt_max_chars,
        )
        return ex or content[: settings.excerpt_max_chars] + OMIT_MARK.rstrip("\n")

    previous = job.result or {}
    done_before = {m["node_id"]: m for m in previous.get("matched_nodes", [])}
    todo = [(nid, sim) for nid, sim in candidates if nid not in done_before]
    docs = {nid: doc_for(nid) for nid, _ in todo}

    # 예산 사전 확인: 판정 도중에 예산이 끊겨 반쪽 결과에 비용만 쓰는 일을 막는다.
    needed = sum(
        estimate_tokens(len(docs[nid])) + _PROMPT_OVERHEAD_TOKENS + _OUTPUT_TOKENS
        for nid, _ in todo
    )
    ensure_budget = getattr(llm, "ensure_budget", None)
    if todo and ensure_budget is not None:
        await ensure_budget(needed)

    async def judge(node_id: str, sim: float) -> MatchedNode:
        node = node_by_id[node_id]
        judgment = await classify_stance(docs[node_id], node, llm, excerpt=excerpt_mode)
        ver = verify_rationale(judgment.rationale_sentences, content)
        stance = judgment.stance
        downgraded = False
        # 지지/반증인데 원문 근거가 하나도 실재하지 않으면 → 약한관련으로 강등
        # (후보로 뜬 만큼 주제 관련은 있으나 증거가 실증되지 않음). 강등 사실은 결과에 남긴다.
        if stance in (Stance.SUPPORTS, Stance.REFUTES) and not ver.kept:
            stance = Stance.WEAK_RELATED
            downgraded = True
        kept, offsets = ver.kept, [list(o) for o in ver.offsets]
        # 불변식: WEAK_RELATED/IRRELEVANT는 rationale/제공유형을 갖지 않는다.
        provided_types = [t.value for t in judgment.provided_evidence_types]
        if stance in (Stance.WEAK_RELATED, Stance.IRRELEVANT):
            kept, offsets, provided_types = [], [], []
        strength = score_strength(judgment.factors, stance)
        log.info(
            "stance content=%s node=%s model=%s→%s dropped=%d repaired=%d via=%s",
            job.content_id, node_id, judgment.stance.value, stance.value,
            ver.dropped, ver.repaired, cands.via.get(node_id, "embedding"),
        )
        return MatchedNode(
            node_id=node_id,
            hypothesis=node["hypothesis"],
            candidate_similarity=round(sim, 4),
            stance=stance,
            rationale_sentences=kept,
            warrant=judgment.warrant,
            provided_evidence_types=provided_types,
            strength=strength,
            rationale_dropped=ver.dropped,
            rationale_repaired=ver.repaired if kept else 0,
            rationale_offsets=offsets,
            downgraded=downgraded,
            candidate_via=cands.via.get(node_id, "embedding"),
        )

    outcomes = await asyncio.gather(*(judge(nid, sim) for nid, sim in todo), return_exceptions=True)

    new_matched: dict[str, MatchedNode] = {}
    failed: list[FailedNode] = []
    first_error: BaseException | None = None
    for (nid, _), out in zip(todo, outcomes):
        if isinstance(out, asyncio.CancelledError):
            raise out
        if isinstance(out, BaseException):
            first_error = first_error or out
            _s, code, msg, _ = describe(out)
            log.warning("stance 실패 content=%s node=%s code=%s: %s", job.content_id, nid, code, out)
            failed.append(
                FailedNode(
                    node_id=nid, hypothesis=node_by_id[nid]["hypothesis"], code=code,
                    message=msg[:300],
                )
            )
        else:
            new_matched[nid] = out

    # 한 노드도 판정하지 못했으면 부분 결과가 아니라 실패다(원인 오류를 그대로 올린다).
    if first_error is not None and not new_matched and not done_before:
        raise first_error

    matched: list[MatchedNode] = []
    for nid, _ in candidates:
        if nid in new_matched:
            matched.append(new_matched[nid])
        elif nid in done_before:
            matched.append(MatchedNode.model_validate(done_before[nid]))

    analysis = ContentAnalysis(
        project_id=job.project_id,
        source_url=job.source_url,
        chunk_count=len(chunks),
        candidate_node_ids=[nid for nid, _ in candidates],
        matched_nodes=matched,
        failed_nodes=failed,
        model=getattr(llm, "model", None) or settings.claude_model,
        prompt_version=PROMPT_VERSION,
        analyzed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        content_chars=len(content),
        input_mode="excerpt" if excerpt_mode else "full",
    )
    return analysis.model_dump(mode="json")


async def analyze_content(
    project_id: str,
    content: str,
    llm: LLMProvider,
    embedder: EmbeddingProvider,
    *,
    source_url: str | None = None,
    max_candidates: int = 5,
    force: bool = False,
) -> dict[str, Any]:
    """접수부터 실행까지 한 번에 (MCP 툴, 평가 스크립트, 동기 REST가 쓴다)."""
    job = await start_analysis(project_id, content, source_url=source_url, force=force)
    if job.kind == "reuse":
        return {**(job.result or {}), "content_id": job.content_id, "status": "done", "reused": True}
    if job.kind == "busy":
        raise AnalysisInProgressError(job.content_id)
    return await run_analysis(job, llm, embedder, max_candidates=max_candidates)
