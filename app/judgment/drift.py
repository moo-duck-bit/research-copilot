"""check_goal_drift: 리서치가 브리프에서 벗어나는지 / 어느 목표가 방치됐는지 판정.

판정은 결정론적 코드(list_collected 출력 기반), 서술(narrative)만 Claude가 생성한다.
확증편향 방지: REFUTES 자료도 목표 기여(on-goal)로 카운트한다.
"""
from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from app.core.config import settings
from app.judgment.collected import list_collected
from app.llm.base import LLMError, LLMProvider
from app.prompts.drift import DRIFT_SYSTEM, build_drift_user
from app.schemas.drift import (
    BranchStatus,
    DriftNarrative,
    DriftSignal,
    GoalDriftResult,
    NeglectedGoal,
)

_TOOL_NAME = "write_drift_narrative"


class TreeStructureError(ValueError):
    """parent_id 순환이나 존재하지 않는 부모 때문에 루트를 정할 수 없을 때."""


def root_of(node_id: str, parent_of: dict[str, str | None]) -> str:
    """parent_id를 따라 올라가 소속 1레벨(루트) node_id를 반환한다.

    node_id 문자열 모양("2.1" → "2")에 기대지 않고 실제 부모 관계만 따른다.
    순환이 있거나 부모가 트리에 없으면 TreeStructureError.
    """
    seen: list[str] = []
    cur = node_id
    while True:
        if cur in seen:
            raise TreeStructureError(f"parent_id 순환: {' -> '.join([*seen, cur])}")
        if cur not in parent_of:
            raise TreeStructureError(f"노드 {cur}가 트리에 없다 (시작: {node_id}).")
        seen.append(cur)
        parent = parent_of[cur]
        if not parent:
            return cur
        cur = parent


def _sanitize(text: str) -> str:
    """모델이 값에 흘린 XML-ish 태그(</narrative>, </invoke> 등)를 제거한다."""
    return re.sub(r"</?[a-zA-Z_][\w-]*>", "", text).strip()


def _severity(drifting: bool, ratio: float) -> str:
    if not drifting:
        return "none"
    if ratio < 0.6:
        return "mild"
    if ratio < 0.8:
        return "moderate"
    return "high"


async def _write_narrative(facts: dict, llm: LLMProvider) -> str:
    tool = {
        "name": _TOOL_NAME,
        "description": "드리프트 판정 사실을 바탕으로 컨설턴트용 서술을 작성한다.",
        "input_schema": DriftNarrative.model_json_schema(),
    }
    try:
        name, data = await llm.extract(
            system=DRIFT_SYSTEM,
            user=build_drift_user(facts),
            tools=[tool],
            tool_choice={"type": "tool", "name": _TOOL_NAME},
        )
        return _sanitize(DriftNarrative.model_validate(data).narrative)
    except (ValidationError, LLMError):
        # LLM 실패 시에도 도구는 동작해야 하므로 결정론적 대체 문안.
        empty = facts.get("wholly_empty_branches") or []
        return (
            f"드리프트 정도: {facts['severity']}. 미분류 비율 {facts['drift_ratio']} "
            f"(무관 {facts['irrelevant_items']}건). 반증 {facts['refutes_items']}건은 정당한 진전으로 "
            f"목표 기여에 포함됨. 통째 빈 브랜치: {empty}. 방치된 목표로 리서치를 되돌릴 것."
        )


async def check_goal_drift(project_id: str, llm: LLMProvider) -> dict[str, Any]:
    view = await list_collected(project_id)
    summary = view["summary"]
    nodes = view["nodes"]

    total = summary["total_content_items"]
    weak = summary["weak_related_items"]
    irr = summary["irrelevant_items"]

    # 확증편향 방지: 지지(SUP) '그리고' 반증(REF) 모두 목표 기여로 카운트
    support_ids: set[str] = set()
    refute_ids: set[str] = set()
    for n in nodes:
        support_ids.update(n["supporting_content_ids"])
        refute_ids.update(n["refuting_content_ids"])
    on_goal_ids = support_ids | refute_ids

    drift_ratio = round((weak + irr) / total, 3) if total else 0.0
    irrelevant_ratio = round(irr / total, 3) if total else 0.0
    threshold = settings.drift_unclassified_threshold
    drifting = total > 0 and drift_ratio >= threshold
    severity = _severity(drifting, drift_ratio)

    signal = DriftSignal(
        drifting=drifting,
        severity=severity,
        drift_ratio=drift_ratio,
        irrelevant_ratio=irrelevant_ratio,
        threshold=threshold,
        total_items=total,
        on_goal_items=len(on_goal_ids),
        supports_items=len(support_ids),
        refutes_items=len(refute_ids),
        weak_related_items=weak,
        irrelevant_items=irr,
    )

    # 1레벨 브랜치별 증거 유무 (SUP 또는 REF 있으면 evidence 있음)
    parent_of = {n["node_id"]: n.get("parent_id") or None for n in nodes}
    level1 = [n for n in nodes if n["level"] == 1]
    branches: list[BranchStatus] = []
    wholly_empty: list[str] = []
    for b in sorted(level1, key=lambda x: x["priority"], reverse=True):
        root = b["node_id"]
        members = [n for n in nodes if root_of(n["node_id"], parent_of) == root]
        ev_count = sum(1 for n in members if n["supports_count"] + n["refutes_count"] > 0)
        has_ev = ev_count > 0
        branches.append(
            BranchStatus(
                branch_node_id=root,
                branch_hypothesis=b["hypothesis"],
                priority=b["priority"],
                has_evidence=has_ev,
                evidence_node_count=ev_count,
                total_nodes=len(members),
            )
        )
        if not has_ev:
            wholly_empty.append(root)

    # 방치된 목표: gap 노드 priority 높은 순
    neglected = sorted(view["gap_nodes"], key=lambda g: g["priority"], reverse=True)
    neglected_goals = [
        NeglectedGoal(
            node_id=g["node_id"],
            hypothesis=g["hypothesis"],
            priority=g["priority"],
            branch=root_of(g["node_id"], parent_of),
        )
        for g in neglected
    ]

    # 다음 방향 제안(결정론): 빈 브랜치 우선 → 고우선 gap
    next_directions: list[str] = []
    for b in branches:
        if not b.has_evidence:
            next_directions.append(
                f"브랜치 {b.branch_node_id}({b.branch_hypothesis[:22]}) 통째 비어있음 → 우선 착수"
            )
    for g in neglected_goals[:3]:
        next_directions.append(
            f"gap node {g.node_id}(p{g.priority}): {g.hypothesis[:28]}"
        )

    facts = {
        "primary_question": view["primary_question"],
        "severity": severity,
        "drifting": drifting,
        "drift_ratio": drift_ratio,
        "irrelevant_items": irr,
        "weak_related_items": weak,
        "on_goal_items": len(on_goal_ids),
        "supports_items": len(support_ids),
        "refutes_items": len(refute_ids),
        "wholly_empty_branches": [
            f"{b.branch_node_id}:{b.branch_hypothesis}" for b in branches if not b.has_evidence
        ],
        "top_neglected_goals": [
            f"{g.node_id}(p{g.priority}):{g.hypothesis}" for g in neglected_goals[:5]
        ],
    }
    narrative = await _write_narrative(facts, llm)

    result = GoalDriftResult(
        project_id=project_id,
        signal=signal,
        neglected_goals=neglected_goals,
        branch_imbalance=branches,
        wholly_empty_branches=wholly_empty,
        next_directions=next_directions,
        narrative=narrative,
    )
    return result.model_dump(mode="json")
