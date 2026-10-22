"""3개 도메인 브리프로 set_project_brief 파이프라인을 실제 호출·저장하고 결과를 출력한다.

컨테이너 안에서 실행:
    docker compose -f docker/docker-compose.yml exec api python scripts/verify_briefs.py
ANTHROPIC_API_KEY 가 컨테이너 환경에 있어야 실제 호출이 된다.
"""
import asyncio
import json

from app.core.db import init_models
from app.judgment.goal_tree import create_project_brief
from app.llm.claude import ClaudeProvider

BRIEFS = {
    "리테일 체인 매출 성장": "국내 중견 리테일 체인의 향후 3년 매출 성장 전략을 도출하고 싶다.",
    "제조업체 원가 절감": "부품 제조업체의 제조 원가를 의미 있게 절감할 방법을 찾고 싶다.",
    "B2B SaaS 신규 시장 진입": "B2B SaaS 기업이 새로운 지역/버티컬 시장에 진입할지 판단하고 싶다.",
}


def _self_check(name: str, payload: dict) -> None:
    if payload["status"] != "complete":
        print(f"  [self-check] status={payload['status']} → 트리 없음")
        return
    nodes = payload["hypothesis_tree"]
    empty_ev = [n["node_id"] for n in nodes if not n["required_evidence"]]
    level1 = [n for n in nodes if n["level"] == 1]
    missing_mece = [n["node_id"] for n in level1 if not n.get("mece_rationale")]
    print(f"  [self-check] 노드 {len(nodes)}개, 1레벨 {len(level1)}개")
    print(f"  [self-check] required_evidence 빈 노드: {empty_ev or '없음(OK)'}")
    print(f"  [self-check] mece_rationale 빠진 1레벨: {missing_mece or '없음(OK)'}")
    print("  [self-check] 1레벨 브랜치:")
    for n in level1:
        print(f"    - {n['node_id']} {n['hypothesis']}  | MECE: {n['mece_rationale']}")


async def main() -> None:
    await init_models()
    provider = ClaudeProvider()
    for name, brief in BRIEFS.items():
        print("\n" + "=" * 70)
        print(f"### {name}")
        print("=" * 70)
        payload = await create_project_brief(brief, provider)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        _self_check(name, payload)


if __name__ == "__main__":
    asyncio.run(main())
