"""골드셋 라벨링 CLI. 자료를 보여주고 (node_id, stance, rationale 문장)을 입력받아
eval/goldset.jsonl 에 기록한다.

컨테이너 안에서 대화형으로 실행: docker compose ... exec api python eval/label_cli.py
(또는 research-copilot/ 에서 `make label`)
"""
import asyncio
import json
import re
from pathlib import Path

from sqlalchemy import text

from app.core.db import SessionLocal

EVAL_DIR = Path(__file__).parent
GOLD_PATH = EVAL_DIR / "goldset.jsonl"
STANCE_MAP = {
    "s": "SUPPORTS",
    "r": "REFUTES",
    "w": "WEAK_RELATED",
    "i": "IRRELEVANT",
}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _save_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _sentences(doc: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。])\s+", doc.strip())
    return [p.strip() for p in parts if p.strip()]


async def _node_hypotheses(project_id: str) -> dict[str, str]:
    async with SessionLocal() as session:
        row = (
            await session.execute(
                text("SELECT tree FROM projects WHERE id=:pid"), {"pid": project_id}
            )
        ).first()
    if not row or not row[0]:
        return {}
    return {n["node_id"]: n["hypothesis"] for n in row[0]["hypothesis_tree"]}


def _upsert(rows: list[dict], rec: dict) -> list[dict]:
    for i, r in enumerate(rows):
        if r["doc_id"] == rec["doc_id"] and r["node_id"] == rec["node_id"]:
            rows[i] = rec
            return rows
    rows.append(rec)
    return rows


async def main() -> None:
    documents = {d["doc_id"]: d for d in _load_jsonl(EVAL_DIR / "documents.jsonl")}
    if not documents:
        print("documents.jsonl 이 비어 있다. 먼저 자료를 추가하라.")
        return

    print("=== 골드셋 라벨링 CLI ===")
    print("자료 목록:")
    for did, d in documents.items():
        print(f"  {did}: {d.get('label','')}")

    while True:
        doc_id = input("\n라벨링할 doc_id (엔터=종료): ").strip()
        if not doc_id:
            break
        if doc_id not in documents:
            print("  없는 doc_id")
            continue
        doc = documents[doc_id]
        sents = _sentences(doc["document"])
        hyps = await _node_hypotheses(doc["project_id"])

        print(f"\n--- 자료 {doc_id} 문장 ---")
        for i, s in enumerate(sents):
            print(f"  [{i}] {s}")
        print("--- 가설 노드 ---")
        for nid, h in hyps.items():
            print(f"  {nid}: {h}")

        node_id = input("\nnode_id: ").strip()
        if node_id and node_id not in hyps:
            print("  (경고) 트리에 없는 node_id — 그래도 기록한다.")
        st = input("stance [s=지지 / r=반증 / w=약한 관련 / i=무관]: ").strip().lower()
        stance = STANCE_MAP.get(st)
        if not stance:
            print("  잘못된 stance. 건너뜀.")
            continue

        rationale: list[str] = []
        if stance in ("SUPPORTS", "REFUTES"):
            idx = input("rationale 문장 번호(쉼표, 예: 0,2): ").strip()
            for tok in idx.split(","):
                tok = tok.strip()
                if tok.isdigit() and int(tok) < len(sents):
                    rationale.append(sents[int(tok)])
        note = input("note(선택): ").strip()

        rows = _load_jsonl(GOLD_PATH)
        rec = {
            "doc_id": doc_id,
            "node_id": node_id,
            "gold_stance": stance,
            "gold_rationale_sentences": rationale,
            "reviewed": True,
            "note": note,
        }
        _save_jsonl(GOLD_PATH, _upsert(rows, rec))
        print(f"  ✓ 저장: {doc_id}/{node_id} = {stance} (rationale {len(rationale)}문장)")

    print("종료.")


if __name__ == "__main__":
    asyncio.run(main())
