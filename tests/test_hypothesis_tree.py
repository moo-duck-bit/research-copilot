"""가설 트리 구조 검증과 node_id 재부여 테스트 (LLM 호출 없음)."""
import pytest
from pydantic import ValidationError

from app.judgment.goal_tree import renumber_node_ids
from app.schemas.hypothesis_tree import HypothesisTree


def node(nid, parent=None, level=1, leaf=True, hyp=None):
    return {
        "node_id": nid,
        "parent_id": parent,
        "level": level,
        "hypothesis": hyp or f"가설 {nid}",
        "sub_questions": ["하위 질문"],
        "required_evidence": [
            {
                "evidence_type": "market_sizing",
                "specificity": "quantitative",
                "recency_required_months": 12,
            }
        ],
        "priority": 3,
        "is_leaf": leaf,
        "mece_rationale": "형제와 겹치지 않음" if level == 1 else None,
    }


def tree(*nodes):
    return {"primary_question": "핵심 질문", "hypothesis_tree": list(nodes)}


def test_valid_tree_passes():
    HypothesisTree.model_validate(
        tree(
            node("1", leaf=False),
            node("1.1", parent="1", level=2),
            node("2"),
        )
    )


def test_cycle_is_rejected():
    with pytest.raises(ValidationError, match="순환"):
        HypothesisTree.model_validate(
            tree(
                node("1"),
                node("2"),
                node("a", parent="b", level=2, leaf=False),
                node("b", parent="a", level=2, leaf=False),
            )
        )


def test_self_parent_is_rejected():
    with pytest.raises(ValidationError, match="순환"):
        HypothesisTree.model_validate(
            tree(node("1"), node("2"), node("x", parent="x", level=2, leaf=False))
        )


def test_level_mismatch_is_rejected():
    with pytest.raises(ValidationError, match="level"):
        HypothesisTree.model_validate(
            tree(
                node("1", leaf=False),
                node("1.1", parent="1", level=3),  # 부모가 1레벨인데 3레벨
                node("2"),
            )
        )


def test_leaf_with_child_is_rejected():
    with pytest.raises(ValidationError, match="is_leaf"):
        HypothesisTree.model_validate(
            tree(
                node("1", leaf=True),  # 자식이 있는데 리프라고 함
                node("1.1", parent="1", level=2),
                node("2"),
            )
        )


def test_non_leaf_without_child_is_rejected():
    with pytest.raises(ValidationError, match="is_leaf"):
        HypothesisTree.model_validate(
            tree(node("1", leaf=False), node("2"))  # 자식이 없는데 리프가 아니라고 함
        )


def test_renumber_fixes_mismatched_prefix():
    # "3.1"인데 실제 부모는 "2" → 재부여 후 "2.1"
    t = HypothesisTree.model_validate(
        tree(
            node("1"),
            node("2", leaf=False),
            node("3.1", parent="2", level=2, hyp="2번 아래 노드"),
        )
    )
    out = renumber_node_ids(t)
    by_hyp = {n.hypothesis: n for n in out.hypothesis_tree}
    assert by_hyp["2번 아래 노드"].node_id == "2.1"
    assert by_hyp["2번 아래 노드"].parent_id == "2"


def test_renumber_follows_given_order_and_depth():
    t = HypothesisTree.model_validate(
        tree(
            node("7", leaf=False, hyp="첫째 루트"),
            node("x", parent="7", level=2, leaf=False, hyp="첫째의 첫 자식"),
            node("4", leaf=False, hyp="둘째 루트"),
            node("y", parent="7", level=2, hyp="첫째의 둘째 자식"),
            node("z", parent="x", level=3, hyp="손자"),
            node("w", parent="4", level=2, hyp="둘째의 자식"),
        )
    )
    out = renumber_node_ids(t)
    got = {n.hypothesis: (n.node_id, n.parent_id) for n in out.hypothesis_tree}
    assert got == {
        "첫째 루트": ("1", None),
        "첫째의 첫 자식": ("1.1", "1"),
        "둘째 루트": ("2", None),
        "첫째의 둘째 자식": ("1.2", "1"),
        "손자": ("1.1.1", "1.1"),
        "둘째의 자식": ("2.1", "2"),
    }
    # 배열 순서와 다른 필드는 그대로
    assert [n.hypothesis for n in out.hypothesis_tree] == [
        n.hypothesis for n in t.hypothesis_tree
    ]


def test_renumber_does_not_mutate_input():
    t = HypothesisTree.model_validate(
        tree(node("9"), node("8", leaf=False), node("q", parent="8", level=2))
    )
    before = t.model_dump()
    renumber_node_ids(t)
    assert t.model_dump() == before
