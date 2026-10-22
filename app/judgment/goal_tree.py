"""goal → 가설 트리 분해 + 저장 (판단 로직, 서버 내부 캡슐화).

tool-use로 강제된 JSON을 Pydantic으로 검증하고, 실패 시 1회 재시도한다.
그래도 실패면 명확한 에러를 던진다.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from pydantic import ValidationError

from app.core.config import settings
from app.core.db import SessionLocal
from app.core.errors import AppError
from app.core.text_safety import sanitize_text
from app.llm.base import LLMError, LLMProvider
from app.models.project import Project
from app.prompts.goal_tree import GOAL_TREE_SYSTEM, build_user_prompt
from app.schemas.hypothesis_tree import ClarificationRequest, HypothesisTree

_PROPOSE_TOOL_NAME = "propose_hypothesis_tree"
_CLARIFY_TOOL_NAME = "request_clarification"


class BriefDecompositionError(LLMError):
    """스키마 검증이 재시도 후에도 실패했을 때."""

    code = "llm_schema_invalid"


def validate_goal(raw: str) -> str:
    goal = sanitize_text(raw or "")
    if not goal:
        raise AppError(422, "goal_empty", "목표(goal)가 비어 있다.")
    if len(goal) > settings.max_goal_chars:
        raise AppError(
            413, "goal_too_long",
            f"목표가 너무 길다({len(goal):,}자, 최대 {settings.max_goal_chars:,}자). 핵심 질문 위주로 줄여라.",
        )
    return goal


def renumber_node_ids(tree: HypothesisTree) -> HypothesisTree:
    """검증된 트리의 node_id를 서버가 구조에 맞게 다시 매긴다 (순수 함수).

    Claude가 준 node_id는 신뢰하지 않고 parent_id 관계만 믿는다.
    - 1레벨 노드: 배열에 나온 순서대로 "1", "2", ...
    - 자식 노드: 부모의 새 번호 + ".1", ".2", ... (형제끼리는 배열 순서)
    - parent_id도 새 번호로 바꾼다. 노드 배열 순서와 나머지 필드는 그대로 둔다.
    입력은 순환·부모 누락이 없는 검증된 트리라고 가정하고, 결과를 다시 검증해 반환한다.
    """
    nodes = tree.hypothesis_tree
    roots: list[int] = []
    children: dict[str, list[int]] = defaultdict(list)
    for i, n in enumerate(nodes):
        if n.parent_id:
            children[n.parent_id].append(i)
        else:
            roots.append(i)

    new_id: dict[int, str] = {}

    def assign(i: int, nid: str) -> None:
        new_id[i] = nid
        for k, child in enumerate(children[nodes[i].node_id], start=1):
            assign(child, f"{nid}.{k}")

    for k, root in enumerate(roots, start=1):
        assign(root, str(k))

    old_to_new = {nodes[i].node_id: nid for i, nid in new_id.items()}
    renumbered = []
    for i, n in enumerate(nodes):
        d = n.model_dump(mode="json")
        d["node_id"] = new_id[i]
        d["parent_id"] = old_to_new[n.parent_id] if n.parent_id else None
        renumbered.append(d)

    return HypothesisTree.model_validate(
        {"primary_question": tree.primary_question, "hypothesis_tree": renumbered}
    )


def _tools() -> list[dict[str, Any]]:
    return [
        {
            "name": _PROPOSE_TOOL_NAME,
            "description": (
                "브리프가 충분히 구체적일 때 잠정 가설 트리를 제안한다. 기본 선택지."
            ),
            "input_schema": HypothesisTree.model_json_schema(),
        },
        {
            "name": _CLARIFY_TOOL_NAME,
            "description": (
                "브리프가 너무 모호해 어떤 트리도 무의미할 때에 한해, "
                "명확화 질문을 정확히 1개만 요청한다."
            ),
            "input_schema": ClarificationRequest.model_json_schema(),
        },
    ]


async def decompose_brief(
    goal: str, provider: LLMProvider
) -> HypothesisTree | ClarificationRequest:
    """브리프를 트리 또는 명확화 질문으로 분해한다. 검증 실패 시 1회 재시도."""
    tools = _tools()
    correction: str | None = None
    last_error: Exception | None = None

    for _attempt in range(2):  # 최초 1회 + 재시도 1회
        user = build_user_prompt(goal, correction)
        name, data = await provider.extract(
            system=GOAL_TREE_SYSTEM, user=user, tools=tools
        )
        try:
            if name == _CLARIFY_TOOL_NAME:
                return ClarificationRequest.model_validate(data)
            if name == _PROPOSE_TOOL_NAME:
                return HypothesisTree.model_validate(data)
            raise ValueError(f"예상치 못한 도구 호출: {name}")
        except (ValidationError, ValueError) as exc:
            last_error = exc
            correction = str(exc)

    raise BriefDecompositionError(
        f"스키마 검증이 재시도 후에도 실패했다: {last_error}"
    )


async def create_project_brief(goal: str, provider: LLMProvider) -> dict[str, Any]:
    """브리프를 분해하고 DB에 저장한다. 저장 결과(project_id 포함)를 반환한다.

    - 명확화가 필요하면: 저장하지 않고 질문만 반환(왕복 최소화).
    - 트리가 나오면: projects 행을 만들고 project_id를 발급한다.
    """
    result = await decompose_brief(goal, provider)

    if isinstance(result, ClarificationRequest):
        return {
            "status": "needs_clarification",
            "project_id": None,
            "clarification_question": result.clarification_question,
        }

    # 검증 통과 직후, 저장 전에 node_id를 구조 기준으로 다시 매긴다.
    result = renumber_node_ids(result)
    tree_json = result.model_dump(mode="json")
    async with SessionLocal() as session:
        project = Project(
            brief=goal,
            status="complete",
            primary_question=result.primary_question,
            tree=tree_json,
        )
        session.add(project)
        await session.commit()
        project_id = str(project.id)

    return {
        "status": "complete",
        "project_id": project_id,
        "primary_question": result.primary_question,
        "hypothesis_tree": tree_json["hypothesis_tree"],
    }
