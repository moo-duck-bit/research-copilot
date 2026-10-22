"""리테일 매출성장 트리를 대상으로 성격이 다른 자료 5개를 analyze_content로 판정한다.

컨테이너 안에서 실행(ANTHROPIC_API_KEY + VOYAGE_API_KEY 필요):
    docker compose -f docker/docker-compose.yml exec -T api python scripts/verify_analyze.py
"""
import asyncio
import json

from sqlalchemy import text

from app.core.db import SessionLocal, init_models
from app.judgment.analyze import analyze_content
from app.judgment.goal_tree import create_project_brief
from app.llm.claude import ClaudeProvider
from app.llm.embeddings import VoyageEmbeddingProvider

RETAIL_BRIEF = "국내 중견 리테일 체인의 향후 3년 매출 성장 전략을 도출하고 싶다."

# 5개 자료 — 각기 다른 성격. (case, expected, 자료)
DOCS = [
    (
        "① 명백한 지지 (온라인/옴니채널 가설)",
        "SUPPORTS",
        "당사 자체 온라인몰의 매출 비중은 2022년 8%에서 2024년 19%로 상승했다. "
        "같은 기간 전사 매출 성장분의 약 70%가 온라인 채널에서 발생했다. "
        "특히 모바일 앱을 통한 재구매율이 오프라인 단독 고객보다 2.3배 높게 나타났다.",
    ),
    (
        "② 명백한 반증 (신규 출점→매출 증가 가설)",
        "REFUTES",
        "2022년 이후 개점한 신규 매장 12곳 중 9곳이 3년차에도 손익분기점에 도달하지 못했다. "
        "신규점 매출의 상당 부분은 반경 3km 내 기존 매장에서 이전된 것으로 분석됐다. "
        "그 결과 신규 출점에 따른 회사 전체 순매출 증가 효과는 사실상 없었다.",
    ),
    (
        "③ 함정: 주제만 겹침 (업계 일반론, 이 회사 증거 아님)",
        "WEAK_RELATED",
        "국내 리테일 시장은 최근 옴니채널과 이커머스 중심으로 빠르게 재편되고 있다. "
        "업계 전문가들은 앞으로 리테일러의 성패가 디지털 전환 역량에 달려 있다고 입을 모은다. "
        "여러 성공 사례가 회자되지만 구체적 성과는 기업마다 크게 다르다.",
    ),
    (
        "④ 애매한 자료 (멤버십, 신호 불명확)",
        "WEAK_RELATED",
        "당사 멤버십 회원 수는 전년 대비 소폭 증가했다. "
        "다만 멤버십 고객의 객단가가 비회원보다 유의미하게 높은지는 아직 명확하지 않다. "
        "일부 매장에서는 긍정적 신호가 있었으나 전사적으로 일관된 패턴은 확인되지 않았다.",
    ),
    (
        "⑤ 완전 무관 (우주 탐사 뉴스)",
        "IRRELEVANT",
        "NASA는 아르테미스 III 미션에서 2027년 우주비행사를 달 남극에 착륙시킬 계획이라고 발표했다. "
        "이번 미션에는 차세대 우주복과 새로 개발한 착륙선이 투입된다. "
        "유럽우주국도 관련 심우주 통신 기술을 공동 개발 중이다.",
    ),
]


async def _get_retail_project_id() -> str:
    async with SessionLocal() as session:
        row = (
            await session.execute(
                text(
                    "SELECT id FROM projects WHERE brief ILIKE '%리테일%'"
                    " OR primary_question ILIKE '%리테일%' ORDER BY created_at LIMIT 1"
                )
            )
        ).first()
    if row:
        return str(row[0])
    # 없으면 새로 생성(2단계 트리 재현)
    res = await create_project_brief(RETAIL_BRIEF, ClaudeProvider())
    return res["project_id"]


async def main() -> None:
    await init_models()
    pid = await _get_retail_project_id()
    print(f"대상 project_id = {pid}\n")

    llm = ClaudeProvider()
    embedder = VoyageEmbeddingProvider()

    for label, expected, doc in DOCS:
        print("=" * 74)
        print(f"### {label}   (기대: {expected})")
        print("=" * 74)
        result = await analyze_content(pid, doc, llm, embedder)
        # 후보 유사도와 판정을 요약 출력
        for m in result["matched_nodes"]:
            print(
                f"  [{m['stance']:<16}] node {m['node_id']:<5} sim={m['candidate_similarity']:.3f}"
                f" strength={m['strength']['score']:.3f} dropped={m['rationale_dropped']}"
            )
            print(f"      가설: {m['hypothesis']}")
            print(f"      rationale(원문발췌): {m['rationale_sentences']}")
            print(f"      warrant: {m['warrant']}")
        # 전체 JSON도 남긴다
        print("  --- full JSON ---")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print()


if __name__ == "__main__":
    asyncio.run(main())
