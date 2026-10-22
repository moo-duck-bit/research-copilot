"""list_collected 검증: 리테일 프로젝트의 content_items를 새 스키마로 재적재한 뒤
목표 충족 뷰를 출력한다.

컨테이너 안: docker compose ... exec -T api python scripts/verify_collected.py
"""
import asyncio
import json

from sqlalchemy import text

from app.core.db import SessionLocal, init_models
from app.judgment.analyze import analyze_content
from app.judgment.collected import list_collected
from app.llm.claude import ClaudeProvider
from app.llm.embeddings import VoyageEmbeddingProvider

DOCS = [
    ("①지지", "당사 자체 온라인몰의 매출 비중은 2022년 8%에서 2024년 19%로 상승했다. 같은 기간 전사 매출 성장분의 약 70%가 온라인 채널에서 발생했다. 특히 모바일 앱을 통한 재구매율이 오프라인 단독 고객보다 2.3배 높게 나타났다."),
    ("②반증", "2022년 이후 개점한 신규 매장 12곳 중 9곳이 3년차에도 손익분기점에 도달하지 못했다. 신규점 매출의 상당 부분은 반경 3km 내 기존 매장에서 이전된 것으로 분석됐다. 그 결과 신규 출점에 따른 회사 전체 순매출 증가 효과는 사실상 없었다."),
    ("③함정", "국내 리테일 시장은 최근 옴니채널과 이커머스 중심으로 빠르게 재편되고 있다. 업계 전문가들은 앞으로 리테일러의 성패가 디지털 전환 역량에 달려 있다고 입을 모은다. 여러 성공 사례가 회자되지만 구체적 성과는 기업마다 크게 다르다."),
    ("④애매", "당사 멤버십 회원 수는 전년 대비 소폭 증가했다. 다만 멤버십 고객의 객단가가 비회원보다 유의미하게 높은지는 아직 명확하지 않다. 일부 매장에서는 긍정적 신호가 있었으나 전사적으로 일관된 패턴은 확인되지 않았다."),
    ("⑤무관", "NASA는 아르테미스 III 미션에서 2027년 우주비행사를 달 남극에 착륙시킬 계획이라고 발표했다. 이번 미션에는 차세대 우주복과 새로 개발한 착륙선이 투입된다. 유럽우주국도 관련 심우주 통신 기술을 공동 개발 중이다."),
]


async def _retail_pid() -> str:
    async with SessionLocal() as s:
        row = (
            await s.execute(
                text("SELECT id FROM projects WHERE brief ILIKE '%리테일%' ORDER BY created_at LIMIT 1")
            )
        ).first()
    return str(row[0])


async def main() -> None:
    await init_models()
    pid = await _retail_pid()
    print(f"project_id = {pid}")

    # 이전 테스트에서 쌓인 중복 content_items 정리 후 5자료 재적재
    async with SessionLocal() as s:
        await s.execute(text("DELETE FROM content_items WHERE project_id=:pid"), {"pid": pid})
        await s.commit()
    print("기존 content_items 삭제 → 5자료 재분석 중...")

    llm, emb = ClaudeProvider(), VoyageEmbeddingProvider()
    print("\n" + "=" * 70)
    print("### 자료별 스탠스 + strength(재정규화/not_assessed)  [analyze 레벨 검증]")
    print("=" * 70)
    for name, doc in DOCS:
        res = await analyze_content(pid, doc, llm, emb, source_url=f"test://{name}")
        print(f"\n[{name}]")
        for m in res["matched_nodes"]:
            st = m["strength"]
            print(f"  {m['stance']:<13} node {m['node_id']:<6} "
                  f"strength={st['score']} not_assessed={st['not_assessed']} "
                  f"factors={st['factors']}")

    view = list_collected(pid)
    view = await view if asyncio.iscoroutine(view) else view

    print("\n" + "=" * 70)
    print("### list_collected 결과")
    print("=" * 70)
    s = view["summary"]
    print(f"[요약] 전체 노드 {s['total_nodes']} | 지지보유 {s['nodes_with_support']} "
          f"(커버리지 {s['coverage_ratio']}) | 증거보유 {s['nodes_with_any_evidence']}")
    print(f"       자료 {s['total_content_items']}건 = 분류됨 {s['classified_items']} / "
          f"미분류 {s['unclassified_items']} (약한관련 {s['weak_related_items']} / 무관 {s['irrelevant_items']})")

    print("\n[증거가 붙은 노드]")
    for n in view["nodes"]:
        if n["supports_count"] or n["refutes_count"]:
            cov = f"{n['required_evidence_covered']}/{n['required_evidence_total']}"
            print(f"  node {n['node_id']:<6} S={n['supports_count']} R={n['refutes_count']} "
                  f"W={n['weak_related_count']} 강도={n['max_support_strength']} 근거유형충족={cov}  | {n['hypothesis'][:32]}")
            for e in n["evidence_coverage"]:
                mark = "✓" if e["covered"] else "·"
                print(f"       [{mark}] {e['evidence_type']} ({e['specificity']}) provided_by={e['provided_by']}")

    print(f"\n[gap 노드 — 증거 0건] {len(view['gap_nodes'])}개")
    for g in view["gap_nodes"]:
        print(f"  node {g['node_id']:<6} p{g['priority']} | {g['hypothesis'][:44]}")

    print(f"\n[약한관련 자료 — synthesis 후보] {len(view['weak_related'])}건")
    for u in view["weak_related"]:
        print(f"  {u['source_url']}  약한관련노드={u['weak_related_node_ids']} best=node {u['best_node_id']}(sim {u['best_similarity']})")
        print(f"      preview: {u['preview']}")

    print(f"\n[무관 자료 — IRRELEVANT] {len(view['irrelevant'])}건")
    for u in view["irrelevant"]:
        print(f"  {u['source_url']}  best=node {u['best_node_id']}(sim {u['best_similarity']})")
        print(f"      preview: {u['preview']}")

    print("\n--- full JSON ---")
    print(json.dumps(view, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
