"""list_collected: 수집·분석된 자료를 목표(가설 트리) 충족 현황으로 집계한다.

LLM 호출 없음 — content_items(저장된 analyze_content 결과)를 읽어 뷰를 구성한다.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import text

from app.core.db import SessionLocal
from app.core.errors import ProjectNotFoundError, parse_project_id
from app.schemas.collected import (
    CollectedSummary,
    CollectedView,
    EvidenceCoverage,
    GapNode,
    NodeStatus,
    UnclassifiedItem,
)


async def list_collected(project_id: str) -> dict[str, Any]:
    project_id = parse_project_id(project_id)
    async with SessionLocal() as session:
        prow = (
            await session.execute(
                text("SELECT primary_question, tree FROM projects WHERE id=:pid"),
                {"pid": project_id},
            )
        ).first()
        if not prow or not prow[1]:
            raise ProjectNotFoundError(f"project_id={project_id} 없음.")
        primary_question, tree = prow[0], prow[1]

        items = (
            await session.execute(
                text(
                    "SELECT id, source_url, content, result FROM content_items"
                    " WHERE project_id=:pid AND status='done' ORDER BY created_at"
                ),
                {"pid": project_id},
            )
        ).all()

    nodes = tree["hypothesis_tree"]

    # 노드별 누적기 초기화
    acc: dict[str, dict[str, Any]] = {
        n["node_id"]: {
            "supports": [],
            "refutes": [],
            "weak": [],
            "max_strength": 0.0,
            "provided_by_type": {},  # evidence_type -> set(content_id)
        }
        for n in nodes
    }

    weak_related_items: list[UnclassifiedItem] = []
    irrelevant_items: list[UnclassifiedItem] = []

    for content_id, source_url, content, result in items:
        content_id = str(content_id)
        matched = result.get("matched_nodes", [])
        item_classified = False
        weak_nodes: list[str] = []
        best_node_id, best_sim = None, -1.0

        for mn in matched:
            nid = mn["node_id"]
            sim = mn.get("candidate_similarity", 0.0)
            if sim > best_sim:
                best_sim, best_node_id = sim, nid
            if nid not in acc:
                continue
            stance = mn["stance"]
            if stance == "SUPPORTS":
                item_classified = True
                acc[nid]["supports"].append(content_id)
                acc[nid]["max_strength"] = max(
                    acc[nid]["max_strength"], mn.get("strength", {}).get("score", 0.0)
                )
                for t in mn.get("provided_evidence_types", []):
                    acc[nid]["provided_by_type"].setdefault(t, set()).add(content_id)
            elif stance == "REFUTES":
                item_classified = True
                acc[nid]["refutes"].append(content_id)
                for t in mn.get("provided_evidence_types", []):
                    acc[nid]["provided_by_type"].setdefault(t, set()).add(content_id)
            elif stance == "WEAK_RELATED":
                acc[nid]["weak"].append(content_id)
                weak_nodes.append(nid)
            # IRRELEVANT는 노드에 아무 것도 적재하지 않는다.

        # 지지/반증이 하나도 없는 자료 = 미분류 → 약한관련 vs 무관으로 분기
        if not item_classified:
            item = UnclassifiedItem(
                content_id=content_id,
                source_url=source_url,
                preview=(content or "")[:80],
                category="WEAK_RELATED" if weak_nodes else "IRRELEVANT",
                best_node_id=best_node_id,
                best_similarity=round(max(best_sim, 0.0), 4),
                weak_related_node_ids=weak_nodes,
            )
            (weak_related_items if weak_nodes else irrelevant_items).append(item)

    # 노드 상태 구성
    node_statuses: list[NodeStatus] = []
    gap_nodes: list[GapNode] = []
    nodes_with_support = 0
    nodes_with_any_evidence = 0

    for n in nodes:
        nid = n["node_id"]
        a = acc[nid]
        req = n.get("required_evidence", [])
        coverage: list[EvidenceCoverage] = []
        covered_count = 0
        for e in req:
            etype = e.get("evidence_type")
            providers = sorted(a["provided_by_type"].get(etype, set()))
            is_covered = bool(providers)
            covered_count += int(is_covered)
            coverage.append(
                EvidenceCoverage(
                    evidence_type=etype,
                    specificity=e.get("specificity", ""),
                    recency_required_months=e.get("recency_required_months", 0),
                    covered=is_covered,
                    provided_by=providers,
                )
            )

        has_support = len(a["supports"]) > 0
        has_any = has_support or len(a["refutes"]) > 0
        nodes_with_support += int(has_support)
        nodes_with_any_evidence += int(has_any)
        if not has_any:
            gap_nodes.append(
                GapNode(node_id=nid, hypothesis=n["hypothesis"], priority=n.get("priority", 0))
            )

        node_statuses.append(
            NodeStatus(
                node_id=nid,
                parent_id=n.get("parent_id") or None,
                hypothesis=n["hypothesis"],
                level=n.get("level", 0),
                priority=n.get("priority", 0),
                supports_count=len(a["supports"]),
                refutes_count=len(a["refutes"]),
                weak_related_count=len(a["weak"]),
                supporting_content_ids=a["supports"],
                refuting_content_ids=a["refutes"],
                weak_related_content_ids=a["weak"],
                max_support_strength=round(a["max_strength"], 3),
                required_evidence_total=len(req),
                required_evidence_covered=covered_count,
                evidence_coverage=coverage,
            )
        )

    total_nodes = len(nodes)
    n_unclassified = len(weak_related_items) + len(irrelevant_items)
    summary = CollectedSummary(
        total_nodes=total_nodes,
        nodes_with_support=nodes_with_support,
        nodes_with_any_evidence=nodes_with_any_evidence,
        coverage_ratio=round(nodes_with_support / total_nodes, 3) if total_nodes else 0.0,
        total_content_items=len(items),
        classified_items=len(items) - n_unclassified,
        unclassified_items=n_unclassified,
        weak_related_items=len(weak_related_items),
        irrelevant_items=len(irrelevant_items),
    )

    view = CollectedView(
        project_id=project_id,
        primary_question=primary_question,
        summary=summary,
        nodes=node_statuses,
        gap_nodes=gap_nodes,
        weak_related=weak_related_items,
        irrelevant=irrelevant_items,
    )
    return view.model_dump(mode="json")
