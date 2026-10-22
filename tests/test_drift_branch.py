"""드리프트의 가지 묶기가 node_id 모양이 아니라 parent_id를 따르는지 테스트 (LLM·DB 없음)."""
import asyncio

import pytest

import app.judgment.drift as drift
from app.judgment.drift import TreeStructureError, root_of
from app.llm.base import LLMError

PARENT_OF = {"1": None, "2": None, "3.1": "2", "3.1.1": "3.1"}


def test_root_of_follows_parent_not_prefix():
    # 이름은 "3.x"지만 실제 부모 사슬은 2번 가지
    assert root_of("3.1", PARENT_OF) == "2"
    assert root_of("3.1.1", PARENT_OF) == "2"
    assert root_of("1", PARENT_OF) == "1"


def test_root_of_raises_on_cycle():
    with pytest.raises(TreeStructureError, match="순환"):
        root_of("a", {"a": "b", "b": "a"})


def test_root_of_raises_on_missing_parent():
    with pytest.raises(TreeStructureError):
        root_of("a", {"a": "없는노드"})


def _node(nid, parent, level, supports=0, priority=3):
    return {
        "node_id": nid,
        "parent_id": parent,
        "hypothesis": f"가설 {nid}",
        "level": level,
        "priority": priority,
        "supports_count": supports,
        "refutes_count": 0,
        "supporting_content_ids": ["c1"] if supports else [],
        "refuting_content_ids": [],
    }


class _FailingLLM:
    """항상 실패해서 결정론적 대체 서술 경로를 타게 한다."""

    async def extract(self, **kwargs):
        raise LLMError("테스트: LLM 호출 안 함")


def test_check_goal_drift_groups_branches_by_parent(monkeypatch):
    view = {
        "primary_question": "핵심 질문",
        "summary": {"total_content_items": 1, "weak_related_items": 0, "irrelevant_items": 0},
        "nodes": [
            _node("1", None, 1, priority=4),
            _node("2", None, 1, priority=5),
            _node("3.1", "2", 2, supports=1),  # 증거 있음, 이름과 달리 2번 가지
            _node("3.2", "2", 2),              # 증거 없음(gap), 역시 2번 가지
        ],
        "gap_nodes": [
            {"node_id": "1", "hypothesis": "가설 1", "priority": 4},
            {"node_id": "2", "hypothesis": "가설 2", "priority": 5},
            {"node_id": "3.2", "hypothesis": "가설 3.2", "priority": 3},
        ],
    }

    async def fake_list_collected(project_id):
        return view

    monkeypatch.setattr(drift, "list_collected", fake_list_collected)
    result = asyncio.run(drift.check_goal_drift("test-project", _FailingLLM()))

    branches = {b["branch_node_id"]: b for b in result["branch_imbalance"]}
    # 2번 가지: 자기 자신 + 3.1 + 3.2, 그중 3.1에 증거
    assert branches["2"]["total_nodes"] == 3
    assert branches["2"]["evidence_node_count"] == 1
    assert branches["2"]["has_evidence"] is True
    # 1번 가지만 통째로 빔 (예전 split 방식이면 2번 가지가 비어 보였다)
    assert result["wholly_empty_branches"] == ["1"]
    # 방치 목표 3.2의 소속 가지도 parent 기준으로 "2"
    goal_branch = {g["node_id"]: g["branch"] for g in result["neglected_goals"]}
    assert goal_branch["3.2"] == "2"
    # LLM 실패 시 대체 서술이 채워진다
    assert result["narrative"]
