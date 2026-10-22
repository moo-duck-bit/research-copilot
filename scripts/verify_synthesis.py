"""find_synthesis_candidates 검증 (리테일 5-자료 프로젝트).

컨테이너 안: docker compose ... exec -T api python scripts/verify_synthesis.py
"""
import asyncio
import json

from sqlalchemy import text

from app.core.db import SessionLocal, init_models
from app.judgment.analyze import analyze_content
from app.judgment.synthesis import find_synthesis_candidates
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
        row = (await s.execute(
            text("SELECT id FROM projects WHERE brief ILIKE '%리테일%' ORDER BY created_at LIMIT 1")
        )).first()
    return str(row[0])


async def _ensure_docs(pid: str) -> dict[str, str]:
    """content_items(source_url→id) 확보. 5건이 없으면 재적재."""
    async with SessionLocal() as s:
        rows = (await s.execute(
            text("SELECT id, source_url FROM content_items WHERE project_id=:pid"), {"pid": pid}
        )).all()
    have = {su: str(i) for i, su in rows}
    if len({k for k in have if k and k.startswith("test://")}) >= 5:
        return have

    print("content_items 5건 부재 → 재적재 중...")
    async with SessionLocal() as s:
        await s.execute(text("DELETE FROM content_items WHERE project_id=:pid"), {"pid": pid})
        await s.commit()
    llm, emb = ClaudeProvider(), VoyageEmbeddingProvider()
    for name, doc in DOCS:
        await analyze_content(pid, doc, llm, emb, source_url=f"test://{name}")
        print(f"  · {name} 재적재")
    async with SessionLocal() as s:
        rows = (await s.execute(
            text("SELECT id, source_url FROM content_items WHERE project_id=:pid"), {"pid": pid}
        )).all()
    return {su: str(i) for i, su in rows}


async def main() -> None:
    await init_models()
    pid = await _retail_pid()
    url_to_id = await _ensure_docs(pid)
    id_to_url = {v: k for k, v in url_to_id.items()}
    trap_id = url_to_id.get("test://③함정")
    space_id = url_to_id.get("test://⑤무관")

    print(f"\nproject_id = {pid}")
    print("=" * 72)
    result = await find_synthesis_candidates(pid, ClaudeProvider())

    print(f"열거된 후보 연결: {result['considered_links']} | "
          f"warrant 게이트 탈락: {result['dropped_by_warrant_gate']} | "
          f"truncated: {result['truncated_candidates']}")
    print(f"최종 제안: {len(result['proposals'])}건\n")

    def label(cid: str) -> str:
        return id_to_url.get(cid, cid[:8])

    for p in result["proposals"]:
        srcs = ", ".join(label(c) for c in p["content_ids"])
        print(f"[{p['kind']:<12}] node {p['target_node_id']:<6} conf={p['confidence']} "
              f"채움={p['fills_evidence_types']}  자료=({srcs})")
        print(f"    가설: {p['target_hypothesis'][:44]}")
        print(f"    warrant: {p['warrant'][:220]}")

    # --- 검증 체크 ---
    print("\n" + "=" * 72)
    print("### 검증 체크")
    to_2_1_from_trap = [
        p for p in result["proposals"]
        if p["target_node_id"] == "2.1" and trap_id in p["content_ids"]
    ]
    print(f"1) ③함정 자료가 node 2.1 제안에 포함? "
          f"{'✅ 예' if to_2_1_from_trap else '❌ 아니오'} "
          f"({[p['fills_evidence_types'] for p in to_2_1_from_trap]})")

    space_in_any = [p for p in result["proposals"] if space_id in p["content_ids"]]
    print(f"2) ⑤무관(우주) 자료가 어떤 제안에도 안 낌? "
          f"{'✅ 예(제안 0건)' if not space_in_any else f'❌ 아니오({len(space_in_any)}건)'}")

    print(f"3) warrant 게이트로 걸러진 근거없는 연결: {result['dropped_by_warrant_gate']}건 "
          f"(> 0이면 게이트 작동)")

    print("\n--- full JSON ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
