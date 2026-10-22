"""FastAPI 앱 엔트리포인트. FastMCP 서버를 같은 코드베이스/프로세스에서 함께 띄운다."""
import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router as api_router
from app.core import tasks
from app.core.config import settings
from app.core.db import check_db, engine, init_models
from app.core.errors import AppError, describe
from app.llm.base import LLMError
from app.mcp.server import mcp

# FastMCP를 ASGI 앱으로 변환해 FastAPI에 마운트한다 (streamable HTTP, /mcp).
# 무상태 모드: 세션을 파드 메모리에 두지 않아, 어느 파드로 요청이 가도 처리된다.
# json_response: SSE 대신 단일 JSON 응답(로드밸런서·프록시 뒤에서 단순하다).
mcp_app = mcp.http_app(
    path="/mcp",
    stateless_http=settings.mcp_stateless_http,
    json_response=settings.mcp_stateless_http,
)


log = logging.getLogger("app.startup")


async def _init_models_with_retry(max_wait: float) -> None:
    """DB가 늦게 떠도 바로 죽지 않고, max_wait초까지 백오프로 재시도한다."""
    deadline = time.monotonic() + max_wait
    delay = 1.0
    while True:
        try:
            await init_models()
            return
        except Exception as exc:  # noqa: BLE001
            if time.monotonic() + delay > deadline:
                raise
            log.warning("스키마 준비 실패, %.0f초 뒤 재시도: %s", delay, exc)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 5.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # FastMCP 세션 매니저 lifespan을 그대로 위임받아 함께 기동한다.
    async with mcp_app.lifespan(app):
        await _init_models_with_retry(settings.db_init_max_wait_seconds)  # 스키마 준비
        yield
        # 종료 신호: 진행 중인 분석을 마저 끝낼 시간을 준다(롤링 배포 중 결과 유실 방지).
        await tasks.drain(settings.shutdown_drain_seconds)
    await engine.dispose()


app = FastAPI(title="AI Research Copilot", version="0.1.0", lifespan=lifespan)

# 웹은 같은 출처(/api)로 부르므로 기본은 CORS를 열지 않는다. 필요할 때만 설정으로 연다.
if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# 웹앱용 REST 라우트
app.include_router(api_router)

# MCP 서버 마운트: FastAPI와 FastMCP가 같은 프로세스에서 뜬다.
app.mount("/mcp-server", mcp_app)


# --- 오류를 원인별 code가 붙은 HTTP 응답으로 (app/core/errors.py) ---
def error_response(exc: BaseException) -> JSONResponse:
    status, code, message, extra = describe(exc)
    headers = {}
    if "retry_after" in extra:
        headers["Retry-After"] = str(extra["retry_after"])
    return JSONResponse({"detail": message, "code": code, **extra}, status_code=status, headers=headers)


@app.exception_handler(AppError)
async def _app_error(_: Request, exc: AppError) -> JSONResponse:
    return error_response(exc)


@app.exception_handler(LLMError)
async def _llm_error(_: Request, exc: LLMError) -> JSONResponse:
    return error_response(exc)


# --- 헬스 체크 ---
@app.get("/livez")
async def livez() -> JSONResponse:
    """liveness: 프로세스가 응답하는지만 본다. DB를 보지 않는다.

    DB가 느려졌다고 정상 파드까지 재시작되는 크래시 루프를 막기 위해서다.
    """
    return JSONResponse({"status": "alive"})


async def _db_reachable(timeout: float) -> bool:
    """제한 시간 안에 DB가 답하면 True.

    asyncio.wait_for는 시간이 지나면 작업을 취소하고 그 취소가 끝날 때까지 기다린다.
    DB가 멈춰 있으면 연결 정리까지 막혀 probe가 매달리므로, 취소만 걸어두고 바로 돌아온다.
    """
    task = asyncio.ensure_future(check_db())
    # 늦게 끝난 작업의 예외가 '처리 안 된 예외'로 로그에 남지 않게 회수한다.
    task.add_done_callback(lambda t: t.cancelled() or t.exception())
    done, _ = await asyncio.wait({task}, timeout=timeout)
    if not done:
        task.cancel()
        return False
    return not task.cancelled() and task.exception() is None


async def _readiness() -> JSONResponse:
    db_ok = await _db_reachable(settings.readiness_db_timeout_seconds)
    payload = {
        "status": "ok" if db_ok else "degraded",
        "database": "connected" if db_ok else "unavailable",
    }
    return JSONResponse(payload, status_code=200 if db_ok else 503)


@app.get("/readyz")
async def readyz() -> JSONResponse:
    """readiness: DB까지 확인한다. 실패하면 트래픽만 빠지고 파드는 재시작되지 않는다."""
    return await _readiness()


@app.get("/health")
async def health() -> JSONResponse:
    """이전 호환용. readiness와 같다."""
    return await _readiness()
