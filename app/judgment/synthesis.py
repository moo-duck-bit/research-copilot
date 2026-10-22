"""find_synthesis_candidates: 증거요건 충족 매트릭스의 조합 탐색(결정론) + 연결 근거 게이트(LLM).

- 매트릭스/후보 열거는 결정론적 코드.
- '이 자료가 정말 이 유형을 제공/기여하는지'만 Claude 게이트로 확인하고, 근거(warrant)를
  못 대는 연결은 버린다(warrant 게이트).
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from pydantic import ValidationError
from sqlalchemy import text

from app.core.db import SessionLocal
from app.judgment.collected import list_collected
from app.llm.base import LLMProvider
from app.prompts.synthesis import SYNTHESIS_GATE_SYSTEM, build_gate_user
from app.schemas.synthesis import SynthesisLinkCheck, SynthesisProposal, SynthesisResult

_TOOL_NAME = "confirm_synthesis_link"
_CONF_THRESHOLD = 0.25   # 게이트 통과 최소 확신도
_MAX_CANDIDATES = 20     # LLM 게이트 상한 (초과분은 truncated로 보고)


def _tool() -> dict:
    return {
        "name": _TOOL_NAME,
        "description": "자료가 노드의 특정 증거유형 요건에 기여하는지와 그 근거를 판정한다.",
        "input_schema": SynthesisLinkCheck.model_json_schema(),
    }


async def _confirm_link(
    hypothesis: str, evidence: dict, doc_content: str, doc_role: str, mode: str, llm: LLMProvider
) -> SynthesisLinkCheck:
    tools = [_tool()]
    choice = {"type": "tool", "name": _TOOL_NAME}
    correction: str | None = None
    for _attempt in range(2):
        user = build_gate_user(hypothesis, evidence, doc_content, doc_role, mode)
        if correction:
            user += "\n[재시도] 스키마 오류를 반영해 다시: " + correction
        name, data = await llm.extract(
            system=SYNTHESIS_GATE_SYSTEM, user=user, tools=tools, tool_choice=choice
        )
        try:
            return SynthesisLinkCheck.model_validate(data)
        except ValidationError as exc:
            correction = str(exc)
    # 재시도 후에도 실패하면 보수적으로 '연결 없음' 처리(게이트가 막는다)
    return SynthesisLinkCheck(connected=False, warrant="", confidence=0.0)


async def find_synthesis_candidates(project_id: str, llm: LLMProvider) -> dict[str, Any]:
    view = await list_collected(project_id)
    node_by_id = {n["node_id"]: n for n in view["nodes"]}

    # 노드별 빈 증거요건
    empty_by_node: dict[str, list[dict]] = {}
    for n in view["nodes"]:
        empties = [e for e in n["evidence_coverage"] if not e["covered"]]
        if empties:
            empty_by_node[n["node_id"]] = empties

    # 자료가 (다른 노드에서) 제공한 유형 + 자료가 붙어있는 노드
    doc_provides: dict[str, set[str]] = defaultdict(set)
    doc_attached_nodes: dict[str, set[str]] = defaultdict(set)
    for n in view["nodes"]:
        for e in n["evidence_coverage"]:
            for cid in e["provided_by"]:
                doc_provides[cid].add(e["evidence_type"])
        for cid in n["supporting_content_ids"] + n["refuting_content_ids"]:
            doc_attached_nodes[cid].add(n["node_id"])

    # WEAK_RELATED 자료 → 약하게 관련된 노드들 (IRRELEVANT는 여기 없음 = 구조적 배제)
    weak_nodes_of = {w["content_id"]: w["weak_related_node_ids"] for w in view["weak_related"]}

    # 자료 원문 로드 (게이트 프롬프트용)
    async with SessionLocal() as session:
        rows = (
            await session.execute(
                text(
                    "SELECT id, content FROM content_items"
                    " WHERE project_id=:pid AND status='done'"
                ),
                {"pid": project_id},
            )
        ).all()
    doc_content = {str(i): c for i, c in rows}

    # --- 결정론적 후보 열거 ---
    candidates: list[dict] = []
    seen: set[tuple] = set()

    def _add(nid: str, cid: str, ev: dict, role: str) -> None:
        key = (nid, cid, ev["evidence_type"])
        if key in seen or cid not in doc_content:
            return
        seen.add(key)
        candidates.append({"node_id": nid, "content_id": cid, "evidence": ev, "role": role})

    # (3) 약한관련 승격 후보
    for cid, wnodes in weak_nodes_of.items():
        for nid in wnodes:
            for ev in empty_by_node.get(nid, []):
                _add(nid, cid, ev, "weak")

    # (1) 빈 근거 채우기: 다른 노드에서 유형 T를 제공한 자료 → T가 빈 노드
    for cid, types in doc_provides.items():
        for nid, empties in empty_by_node.items():
            if nid in doc_attached_nodes.get(cid, set()):
                continue
            for ev in empties:
                if ev["evidence_type"] in types:
                    _add(nid, cid, ev, "provider")

    considered = len(candidates)
    truncated = 0
    if considered > _MAX_CANDIDATES:
        candidates.sort(key=lambda c: node_by_id[c["node_id"]].get("priority", 0), reverse=True)
        truncated = considered - _MAX_CANDIDATES
        candidates = candidates[:_MAX_CANDIDATES]

    # --- LLM 게이트 (병렬) ---
    async def _gate(c: dict) -> tuple[dict, SynthesisLinkCheck]:
        node = node_by_id[c["node_id"]]
        if c["role"] == "weak":
            role_txt = "현재 WEAK_RELATED (어느 노드에도 확정 증거로 붙지 않음)."
            mode = "INVESTIGATE"
        else:
            role_txt = (
                "이미 다른 노드에 확정 증거로 붙어 있으며 제공 유형: "
                f"{sorted(doc_provides.get(c['content_id'], []))}."
            )
            mode = "PROVIDE"
        chk = await _confirm_link(
            node["hypothesis"], c["evidence"], doc_content.get(c["content_id"], ""),
            role_txt, mode, llm,
        )
        return c, chk

    gated = await asyncio.gather(*(_gate(c) for c in candidates)) if candidates else []

    # warrant 게이트: connected + warrant 존재 + 확신도 임계 통과만 유지
    kept: list[tuple[dict, SynthesisLinkCheck]] = []
    dropped = 0
    for c, chk in gated:
        if chk.connected and chk.warrant.strip() and chk.confidence >= _CONF_THRESHOLD:
            kept.append((c, chk))
        else:
            dropped += 1

    # --- 조합 조립(결정론) ---
    # (node, cid)별로 묶어 단일 제안 생성
    by_pair: dict[tuple[str, str], list[tuple[dict, SynthesisLinkCheck]]] = defaultdict(list)
    for c, chk in kept:
        by_pair[(c["node_id"], c["content_id"])].append((c, chk))

    proposals: list[SynthesisProposal] = []
    per_node: dict[str, list[SynthesisProposal]] = defaultdict(list)

    for (nid, cid), lst in by_pair.items():
        types = sorted({c["evidence"]["evidence_type"] for c, _ in lst})
        warrant = " / ".join(chk.warrant for _, chk in lst)
        conf = round(sum(chk.confidence for _, chk in lst) / len(lst), 3)
        kind = "weak_promote" if lst[0][0]["role"] == "weak" else "fill_gap"
        p = SynthesisProposal(
            kind=kind,
            target_node_id=nid,
            target_hypothesis=node_by_id[nid]["hypothesis"],
            content_ids=[cid],
            fills_evidence_types=types,
            warrant=warrant,
            confidence=conf,
        )
        proposals.append(p)
        per_node[nid].append(p)

    # (2) 자료 조합: 한 노드에 서로 다른 자료가 서로 다른 빈 유형을 부분 충족하면 결합 제안
    for nid, plist in per_node.items():
        cids = {p.content_ids[0] for p in plist}
        all_types: set[str] = set().union(*[set(p.fills_evidence_types) for p in plist])
        if len(cids) >= 2 and len(all_types) >= 2:
            proposals.append(
                SynthesisProposal(
                    kind="combine",
                    target_node_id=nid,
                    target_hypothesis=node_by_id[nid]["hypothesis"],
                    content_ids=sorted(cids),
                    fills_evidence_types=sorted(all_types),
                    warrant="서로 다른 자료가 이 노드의 서로 다른 빈 증거 유형을 각각 부분 "
                    "충족한다: " + "; ".join(
                        f"{p.content_ids[0][:8]}→{p.fills_evidence_types}" for p in plist
                    ),
                    confidence=round(min(p.confidence for p in plist), 3),
                )
            )

    # 확신도 높은 순 정렬
    proposals.sort(key=lambda p: p.confidence, reverse=True)

    result = SynthesisResult(
        project_id=project_id,
        considered_links=considered,
        dropped_by_warrant_gate=dropped,
        truncated_candidates=truncated,
        proposals=proposals,
    )
    return result.model_dump(mode="json")
