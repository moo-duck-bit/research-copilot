"""웹앱용 얇은 REST 래퍼. 판단 로직은 재사용만 하고 새 로직을 넣지 않는다.

MCP 툴과 동일한 judgment 함수를 그대로 호출한다.
"""
import uuid
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import text

from app.core import tasks
from app.core.config import settings
from app.core.db import SessionLocal
from app.core.errors import AppError, parse_project_id
from app.judgment.analyze import analyze_content, run_analysis, start_analysis
from app.judgment.collected import list_collected
from app.judgment.drift import check_goal_drift
from app.judgment.goal_tree import create_project_brief, validate_goal
from app.judgment.synthesis import find_synthesis_candidates
from app.llm.base import LLMProvider
from app.llm.claude import ClaudeProvider, default_breaker
from app.llm.embeddings import EmbeddingProvider, VoyageEmbeddingProvider
from app.llm.usage import DbUsageLedger, next_reset

router = APIRouter(prefix="/api", tags=["web"])


# 테스트에서 가짜 구현으로 바꿔 끼울 수 있게 의존성으로 둔다.
def get_llm() -> LLMProvider:
    return ClaudeProvider()


def get_embedder() -> EmbeddingProvider:
    return VoyageEmbeddingProvider()


class BriefIn(BaseModel):
    goal: str


class AnalyzeIn(BaseModel):
    project_id: str
    content: str
    source_url: str | None = None
    force: bool = False


@router.get("/usage")
async def get_usage() -> dict[str, Any]:
    """오늘 LLM 토큰 사용량과 예산, 다시 채워지는 시각, 이 파드의 서킷브레이커 상태."""
    snap = await DbUsageLedger(settings.budget_utc_offset_hours).snapshot()
    used = snap["input_tokens"] + snap["output_tokens"]
    budget = settings.daily_token_budget
    return {
        **snap,
        "budget": budget,
        "remaining": None if budget <= 0 else max(0, budget - used),
        "resets_at": next_reset(settings.budget_utc_offset_hours).isoformat(),
        "max_content_chars": settings.max_content_chars,
        "breaker_state_this_pod": default_breaker().state,
        "running_analyses_this_pod": tasks.running_count(),
    }


@router.get("/projects")
async def list_projects() -> list[dict[str, Any]]:
    async with SessionLocal() as s:
        rows = (
            await s.execute(
                text(
                    "SELECT id, primary_question, status, created_at FROM projects"
                    " ORDER BY created_at DESC LIMIT 50"
                )
            )
        ).all()
    return [
        {"project_id": str(i), "primary_question": q, "status": st, "created_at": str(c)}
        for i, q, st, c in rows
    ]


@router.post("/projects")
async def create_brief(body: BriefIn, llm: LLMProvider = Depends(get_llm)) -> dict[str, Any]:
    return await create_project_brief(validate_goal(body.goal), llm)


@router.get("/projects/{project_id}")
async def get_project(project_id: str) -> dict[str, Any]:
    pid = parse_project_id(project_id)
    async with SessionLocal() as s:
        row = (
            await s.execute(
                text("SELECT primary_question, tree, status FROM projects WHERE id=:pid"),
                {"pid": pid},
            )
        ).first()
    if not row or not row[1]:
        raise AppError(404, "project_not_found", "project_id를 찾을 수 없다.")
    return {
        "project_id": pid,
        "primary_question": row[0],
        "hypothesis_tree": row[1]["hypothesis_tree"],
        "status": row[2],
    }


@router.post("/analyze")
async def post_analyze(
    body: AnalyzeIn,
    llm: LLMProvider = Depends(get_llm),
    embedder: EmbeddingProvider = Depends(get_embedder),
) -> dict[str, Any]:
    """동기 분석(끝날 때까지 기다린다). 웹 화면은 아래 /analyze/jobs를 쓴다."""
    return await analyze_content(
        body.project_id, body.content, llm, embedder,
        source_url=body.source_url, force=body.force,
    )


@router.post("/analyze/jobs")
async def post_analyze_job(
    body: AnalyzeIn,
    llm: LLMProvider = Depends(get_llm),
    embedder: EmbeddingProvider = Depends(get_embedder),
):
    """분석을 접수하고 바로 돌려준다(202). 결과는 GET /api/contents/{content_id}로 확인한다.

    같은 자료가 이미 분석돼 있으면 새로 돌리지 않고 저장된 결과를 200으로 돌려준다.
    """
    job = await start_analysis(
        body.project_id, body.content, source_url=body.source_url, force=body.force
    )
    if job.kind == "reuse":
        return {
            "content_id": job.content_id, "status": "done", "reused": True,
            "result": {**(job.result or {}), "content_id": job.content_id},
        }
    if job.kind == "run":
        tasks.spawn(run_analysis(job, llm, embedder), name=f"analyze:{job.content_id}")
    return JSONResponse(
        {"content_id": job.content_id, "status": "processing", "reused": False},
        status_code=202,
    )


_CONTENT_COLS = (
    "SELECT id, project_id, source_url, status, result, error_code, error_message,"
    " created_at, left(content, 160), length(content),"
    " status = 'processing' AND updated_at < now() - make_interval(secs => :stale)"
    " FROM content_items"
)


def _content_view(row) -> dict[str, Any]:
    cid, pid, url, status, result, ecode, emsg, created, preview, chars, stale = row
    cid = str(cid)
    out: dict[str, Any] = {
        "content_id": cid,
        "project_id": str(pid),
        "source_url": url,
        "status": status,
        "created_at": created.isoformat() if created else None,
        "preview": preview,
        "content_chars": chars,
    }
    if stale:  # 파드 재시작 등으로 끊긴 작업. 다시 제출하면 이어서 돌린다.
        out["status"] = "failed"
        ecode, emsg = "analysis_interrupted", "분석이 중간에 끊겼다(서버 재시작 등). 다시 시도하라."
    if out["status"] == "done":
        out["result"] = {**(result or {}), "content_id": cid}
    elif out["status"] == "failed":
        out["error"] = {"code": ecode, "message": emsg}
    return out


@router.get("/contents/{content_id}")
async def get_content(content_id: str) -> dict[str, Any]:
    try:
        cid = str(uuid.UUID(content_id))
    except ValueError:
        raise AppError(404, "content_not_found", "content_id를 찾을 수 없다.") from None
    async with SessionLocal() as s:
        row = (
            await s.execute(
                text(_CONTENT_COLS + " WHERE id=:id"),
                {"id": cid, "stale": settings.analysis_stale_after_seconds},
            )
        ).first()
    if not row:
        raise AppError(404, "content_not_found", "content_id를 찾을 수 없다.")
    return _content_view(row)


@router.get("/projects/{project_id}/contents")
async def list_contents(project_id: str) -> list[dict[str, Any]]:
    """이 프로젝트에 넣은 자료와 분석 결과(최신순). 새로고침해도 결과가 남아 있게 한다."""
    pid = parse_project_id(project_id)
    async with SessionLocal() as s:
        rows = (
            await s.execute(
                text(_CONTENT_COLS + " WHERE project_id=:pid ORDER BY created_at DESC LIMIT 50"),
                {"pid": pid, "stale": settings.analysis_stale_after_seconds},
            )
        ).all()
    return [_content_view(r) for r in rows]


@router.get("/projects/{project_id}/collected")
async def get_collected(project_id: str) -> dict[str, Any]:
    return await list_collected(project_id)


@router.get("/projects/{project_id}/synthesis")
async def get_synthesis(project_id: str, llm: LLMProvider = Depends(get_llm)) -> dict[str, Any]:
    return await find_synthesis_candidates(parse_project_id(project_id), llm)


@router.get("/projects/{project_id}/drift")
async def get_drift(project_id: str, llm: LLMProvider = Depends(get_llm)) -> dict[str, Any]:
    return await check_goal_drift(parse_project_id(project_id), llm)
