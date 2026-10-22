"""check_goal_drift 검증 (리테일 5-자료 프로젝트).

컨테이너 안: docker compose ... exec -T api python scripts/verify_drift.py
"""
import asyncio
import json

from sqlalchemy import text

from app.core.db import SessionLocal, init_models
from app.judgment.drift import check_goal_drift
from app.llm.claude import ClaudeProvider


async def _retail_pid() -> str:
    async with SessionLocal() as s:
        row = (await s.execute(
            text("SELECT id FROM projects WHERE brief ILIKE '%리테일%' ORDER BY created_at LIMIT 1")
        )).first()
    return str(row[0])


async def main() -> None:
    await init_models()
    pid = await _retail_pid()
    result = await check_goal_drift(pid, ClaudeProvider())

    sig = result["signal"]
    print("=" * 72)
    print("### check_goal_drift 결과")
    print("=" * 72)
    print(f"[드리프트] drifting={sig['drifting']} severity={sig['severity']} "
          f"drift_ratio={sig['drift_ratio']} (임계 {sig['threshold']})")
    print(f"          자료 {sig['total_items']}건 = 목표기여(on-goal) {sig['on_goal_items']} "
          f"[지지 {sig['supports_items']} / 반증 {sig['refutes_items']}] "
          f"| 약한관련 {sig['weak_related_items']} / 무관 {sig['irrelevant_items']}")

    print(f"\n[방치된 목표(gap) — priority 높은 순] {len(result['neglected_goals'])}개")
    for g in result["neglected_goals"]:
        print(f"  p{g['priority']} node {g['node_id']:<6} (br {g['branch']}) | {g['hypothesis'][:40]}")

    print("\n[브랜치 불균형]")
    for b in result["branch_imbalance"]:
        mark = "증거있음" if b["has_evidence"] else "★통째 비어있음"
        print(f"  브랜치 {b['branch_node_id']} p{b['priority']} [{mark}] "
              f"{b['evidence_node_count']}/{b['total_nodes']} 노드 | {b['branch_hypothesis'][:30]}")
    print(f"  → 통째 빈 브랜치: {result['wholly_empty_branches']}")

    print("\n[다음 리서치 방향(결정론)]")
    for d in result["next_directions"]:
        print(f"  - {d}")

    print("\n[서술(LLM)]")
    print("  " + result["narrative"])

    # --- 검증 체크 ---
    print("\n" + "=" * 72)
    print("### 검증 체크")
    ng = result["neglected_goals"]
    prios = [g["priority"] for g in ng]
    print(f"1) gap {len(ng)}개가 priority 내림차순? "
          f"{'✅' if prios == sorted(prios, reverse=True) else '❌'} (priorities={prios})")

    empty = set(result["wholly_empty_branches"])
    want_empty = {"3", "4", "5"}
    print(f"2) 상품(3)/고객(4)/외부환경(5) 브랜치 통째 gap 경고? "
          f"{'✅' if want_empty <= empty else '❌'} (통째 빈 브랜치={sorted(empty)})")

    print(f"3) 미분류 3건이 드리프트 신호에 반영? drift_ratio={sig['drift_ratio']} "
          f"(=(약한관련2+무관1)/5=0.6), drifting={sig['drifting']} "
          f"{'✅' if sig['drift_ratio'] == 0.6 and sig['drifting'] else '❌'}")

    print(f"4) ②반증 자료가 목표기여로 카운트(벗어남 오판 방지)? "
          f"refutes_items={sig['refutes_items']}, on_goal={sig['on_goal_items']} "
          f"{'✅ (반증 1건이 on-goal에 포함, 미분류 아님)' if sig['refutes_items'] >= 1 else '❌'}")

    print("\n--- full JSON ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
