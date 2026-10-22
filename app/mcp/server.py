"""FastMCP 서버. MCP 툴의 시그니처/docstring만 정의하고 판단 로직은 app/judgment에 있다.

판단 로직·프롬프트·드리프트 기준은 전부 서버 내부(app/judgment, app/prompts)에 두며,
클라이언트로 노출하지 않는다.

툴 설명은 호스트 모델이 읽고 인자를 채우는 '사용 설명서'다. 호스트 모델이 긴 원문을
요약·번역해서 넘기거나 project_id를 지어내는 일이 잦아, 그러지 말라는 내용을 적어 둔다.
오류는 [code] 메시지 형태의 ToolError로 돌려줘 호스트 모델이 원인을 구분하게 한다.
"""
from collections.abc import Awaitable
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

mcp = FastMCP("ai-research-copilot")


async def _run(coro: Awaitable[dict[str, Any]]) -> dict[str, Any]:
    from app.core.errors import AppError, describe
    from app.llm.base import LLMError

    try:
        return await coro
    except (AppError, LLMError) as exc:
        _status, code, message, extra = describe(exc)
        hint = f" (retry_after={extra['retry_after']}s)" if "retry_after" in extra else ""
        raise ToolError(f"[{code}] {message}{hint}") from exc


@mcp.tool()
async def set_project_brief(goal: str) -> dict[str, Any]:
    """자연어 리서치 목표(goal)를 받아 가설 트리(각 노드에 required_evidence 포함)로
    분해하고 새 프로젝트로 저장한다.

    사람이 리서치의 방향을 정하는 출발점. AI는 목표를 구조화만 보조한다.
    반환: {status, project_id, primary_question, hypothesis_tree} 또는
    {status: 'needs_clarification', clarification_question}.
    반환된 project_id를 이후 다른 툴에 그대로 넘긴다(직접 만들거나 고치지 않는다).
    """
    from app.judgment.goal_tree import create_project_brief, validate_goal
    from app.llm.claude import ClaudeProvider

    async def go() -> dict[str, Any]:
        return await create_project_brief(validate_goal(goal), ClaudeProvider())

    return await _run(go())


@mcp.tool()
async def analyze_content(
    project_id: str, content: str, source_url: str | None = None
) -> dict[str, Any]:
    """수집한 텍스트 자료를 project_id의 가설 트리에 비추어 분석한다.

    content에는 사용자가 준 원문 전체를 글자 그대로 넣는다. 요약·번역·발췌·재작성하지 않는다.
    근거 문장은 이 원문과 글자 단위로 대조해 검증하므로, 바뀐 텍스트에서는 근거가 버려진다.
    URL만 넣으면 거절된다(서버는 URL 내용을 가져오지 않는다). source_url은 출처 기록용이다.
    같은 자료를 다시 넣으면 새로 분석하지 않고 저장된 결과를 돌려준다(reused=true).

    임베딩 유사도는 후보 필터(pgvector)로만 쓰고, 그 위에 스탠스 분류
    (SUPPORTS/REFUTES/WEAK_RELATED/IRRELEVANT) + rationale 문장 추출(원문 발췌) +
    warrant + 근거 강도를 얹어 노드별 관련성 판정을 반환한다. 판정하지 못한 노드는
    failed_nodes에 따로 담긴다(무관으로 간주하지 말 것).
    """
    from app.judgment.analyze import analyze_content as run_analyze
    from app.llm.claude import ClaudeProvider
    from app.llm.embeddings import VoyageEmbeddingProvider

    return await _run(
        run_analyze(
            project_id,
            content,
            ClaudeProvider(),
            VoyageEmbeddingProvider(),
            source_url=source_url,
        )
    )


@mcp.tool()
async def find_synthesis_candidates(project_id: str) -> dict[str, Any]:
    """자료 조합으로 노드의 빈 근거를 채울 수 있는 연결 후보를 제안한다.

    증거요건 충족 매트릭스의 조합 탐색(결정론) + 각 연결의 근거를 Claude로 확인하는
    warrant 게이트. 근거를 못 대는 연결은 제안하지 않는다. 결론은 사람이 내린다.
    """
    from app.core.errors import parse_project_id
    from app.judgment.synthesis import find_synthesis_candidates as run_synth
    from app.llm.claude import ClaudeProvider

    async def go() -> dict[str, Any]:
        return await run_synth(parse_project_id(project_id), ClaudeProvider())

    return await _run(go())


@mcp.tool()
async def check_goal_drift(project_id: str) -> dict[str, Any]:
    """리서치가 브리프에서 벗어나는지 / 어느 목표가 방치됐는지 판정해 되돌림 신호를 준다.

    판정 근거(미분류 비율, gap 노드, 브랜치 불균형)는 결정론적 계산이고 서술만 LLM.
    반증(REFUTES) 자료도 목표 기여로 카운트한다(확증편향 방지).
    """
    from app.core.errors import parse_project_id
    from app.judgment.drift import check_goal_drift as run_drift
    from app.llm.claude import ClaudeProvider

    async def go() -> dict[str, Any]:
        return await run_drift(parse_project_id(project_id), ClaudeProvider())

    return await _run(go())


@mcp.tool()
async def list_collected(project_id: str) -> dict[str, Any]:
    """project_id의 수집·분석 자료 현황을 목표 충족 뷰로 반환한다.

    노드별 지지/반증 건수 + required_evidence 유형별 충족 여부, 아직 어느 목표에도
    기여 못 한 '미분류' 자료, 그리고 커버리지/증거 0건(gap) 노드 요약을 담는다.
    분석이 끝난 자료만 집계한다(처리 중·실패한 자료는 빠진다).
    """
    from app.judgment.collected import list_collected as run_list

    return await _run(run_list(project_id))
