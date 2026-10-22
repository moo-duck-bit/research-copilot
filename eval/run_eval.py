"""analyze_content 품질 eval 하네스.

골드셋(eval/goldset.jsonl)의 (자료, node_id, 정답 stance, 정답 rationale)에 대해
파이프라인 예측을 비교한다. 지표 2개(SciFact 방식):
  1) label-only F1   : stance만 맞으면 정답 (macro-F1, 3-class)
  2) selection+label : rationale 문장까지 정확히 집었는지 (문장 단위 micro-F1,
                       단 stance가 맞을 때만 문장 크레딧 부여 — 더 엄격)
그리고 SUPPORTS/REFUTES/WEAK_RELATED/IRRELEVANT 4x4 혼동행렬.

컨테이너 안에서 실행: docker compose ... exec -T api python eval/run_eval.py
(또는 research-copilot/ 에서 `make eval`)
"""
import asyncio
import json
import re
from collections import defaultdict
from pathlib import Path

from sqlalchemy import text

from app.core.db import SessionLocal, init_models
from app.judgment.analyze import analyze_content
from app.llm.claude import ClaudeProvider
from app.llm.embeddings import VoyageEmbeddingProvider

EVAL_DIR = Path(__file__).parent
STANCES = ["SUPPORTS", "REFUTES", "WEAK_RELATED", "IRRELEVANT"]
_SHORT = {"SUPPORTS": "SUP", "REFUTES": "REF", "WEAK_RELATED": "WEAK", "IRRELEVANT": "IRR"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


async def _resolve_project_id(pid: str) -> str:
    """documents.jsonl의 project_id가 DB에 없으면 '리테일' 브리프로 대체 탐색."""
    async with SessionLocal() as session:
        row = (
            await session.execute(
                text("SELECT id FROM projects WHERE id=:pid"), {"pid": pid}
            )
        ).first()
        if row:
            return pid
        row = (
            await session.execute(
                text(
                    "SELECT id FROM projects WHERE brief ILIKE '%리테일%'"
                    " OR primary_question ILIKE '%리테일%' ORDER BY created_at LIMIT 1"
                )
            )
        ).first()
    if not row:
        raise RuntimeError(
            f"project_id={pid} 도, 리테일 프로젝트도 DB에 없다. 먼저 set_project_brief로 트리를 만들어라."
        )
    return str(row[0])


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def _label_only_f1(pairs: list[tuple[str, str]]) -> tuple[float, dict]:
    """pairs: (gold, pred). class별 F1 + macro-F1."""
    per_class = {}
    for c in STANCES:
        tp = sum(1 for g, p in pairs if g == c and p == c)
        fp = sum(1 for g, p in pairs if g != c and p == c)
        fn = sum(1 for g, p in pairs if g == c and p != c)
        per_class[c] = _prf(tp, fp, fn)
    macro = sum(f for _, _, f in per_class.values()) / len(STANCES)
    return macro, per_class


def _confusion(pairs: list[tuple[str, str]]) -> dict:
    m = {g: {p: 0 for p in STANCES} for g in STANCES}
    for g, p in pairs:
        m[g][p] += 1
    return m


def _print_confusion(m: dict) -> None:
    print("\n혼동행렬 (행=정답, 열=예측)")
    print("          " + "".join(f"{_SHORT[c]:>6}" for c in STANCES))
    for g in STANCES:
        print(f"  {_SHORT[g]:>6} | " + "".join(f"{m[g][c]:>6}" for c in STANCES))


async def main() -> None:
    await init_models()
    documents = {d["doc_id"]: d for d in _load_jsonl(EVAL_DIR / "documents.jsonl")}
    gold = _load_jsonl(EVAL_DIR / "goldset.jsonl")

    bad = [g for g in gold if g.get("gold_stance") not in STANCES]
    if bad:
        rows = ", ".join(f'{g["doc_id"]}/{g["node_id"]}={g.get("gold_stance")}' for g in bad)
        raise SystemExit(
            f"골드셋에 알 수 없는 stance가 있다: {rows}\n"
            f"허용값: {', '.join(STANCES)}. goldset.jsonl을 고친 뒤 다시 실행하라."
        )

    n_unreviewed = sum(1 for g in gold if not g.get("reviewed"))
    if n_unreviewed:
        print(
            f"⚠️  경고: {n_unreviewed}/{len(gold)} 쌍이 reviewed=false (미검토). "
            "시드 라벨은 모델 예측과 동일하므로 검토 전 점수는 순환적(참고용)이다.\n"
        )

    llm, embedder = ClaudeProvider(), VoyageEmbeddingProvider()

    # 자료별 1회 예측 → (doc_id, node_id) -> (stance, rationale)
    preds: dict[tuple[str, str], tuple[str, list[str]]] = {}
    cand_of: dict[str, set[str]] = {}
    dropped = kept = downgraded = 0
    doc_ids = sorted({g["doc_id"] for g in gold})
    for doc_id in doc_ids:
        doc = documents[doc_id]
        pid = await _resolve_project_id(doc["project_id"])
        print(f"· 예측 실행: {doc_id} ({doc.get('label','')})")
        # force: 같은 자료의 저장된 결과를 재사용하지 않고 지금 프롬프트·모델로 다시 판정한다.
        result = await analyze_content(pid, doc["document"], llm, embedder, force=True)
        cand_of[doc_id] = set(result.get("candidate_node_ids", []))
        for f in result.get("failed_nodes", []):
            print(f"  ! 판정 실패 node {f['node_id']}: [{f['code']}] {f['message'][:80]}")
        for mnode in result["matched_nodes"]:
            dropped += mnode.get("rationale_dropped", 0)
            kept += len(mnode.get("rationale_sentences", []))
            downgraded += int(mnode.get("downgraded", False))
            preds[(doc_id, mnode["node_id"])] = (
                mnode["stance"],
                mnode["rationale_sentences"],
            )

    # 채점
    label_pairs: list[tuple[str, str]] = []
    s_tp = s_fp = s_fn = 0
    rows = []
    for g in gold:
        key = (g["doc_id"], g["node_id"])
        # 후보로 안 뜬 노드는 관련도가 낮아 후보에서 빠진 것 = IRRELEVANT/[] 로 간주
        p_stance, p_rat = preds.get(key, ("IRRELEVANT", []))
        g_stance = g["gold_stance"]
        g_rat = g.get("gold_rationale_sentences", [])
        label_pairs.append((g_stance, p_stance))

        g_set = {_norm(x) for x in g_rat}
        p_set = {_norm(x) for x in p_rat}
        if p_stance == g_stance:
            s_tp += len(g_set & p_set)
            s_fp += len(p_set - g_set)
            s_fn += len(g_set - p_set)
        else:  # stance 틀리면 문장 크레딧 없음
            s_fp += len(p_set)
            s_fn += len(g_set)
        rows.append((g["doc_id"], g["node_id"], g_stance, p_stance,
                     "OK" if g_stance == p_stance else "X"))

    print("\n" + "=" * 60)
    print("쌍별 결과 (정답 stance → 예측 stance)")
    for doc_id, node_id, gs, ps, ok in rows:
        print(f"  [{ok:>2}] {doc_id:<5} node {node_id:<6} {gs:>16} → {ps}")

    # [지표 0] 후보 재현율: 정답이 지지/반증인 쌍에서, 그 노드가 1단계 후보에 들었는가.
    # 후보에서 빠지면 판정 기회 자체가 없어 '무관'으로 조용히 처리된다. 판정 F1과 따로 본다.
    ev = [g for g in gold if g["gold_stance"] in ("SUPPORTS", "REFUTES")]
    hit = sum(1 for g in ev if g["node_id"] in cand_of.get(g["doc_id"], set()))
    print("\n" + "=" * 60)
    print("[지표 0] 후보 재현율 (지지/반증 정답 노드가 후보에 든 비율)")
    print(f"  recall = {hit}/{len(ev)} = {hit / len(ev) if ev else 0:.3f}")
    print(
        f"  근거 문장 원문 대조: 통과 {kept}, 탈락 {dropped}, "
        f"근거 없음으로 강등된 판정 {downgraded}"
    )

    macro, per_class = _label_only_f1(label_pairs)
    acc = sum(1 for g, p in label_pairs if g == p) / len(label_pairs)
    print("\n" + "=" * 60)
    print(f"[지표 1] label-only  (stance만)")
    print(f"  accuracy = {acc:.3f}   macro-F1 = {macro:.3f}")
    for c in STANCES:
        p, r, f = per_class[c]
        print(f"    {c:<16} P={p:.3f} R={r:.3f} F1={f:.3f}")

    sp, sr, sf = _prf(s_tp, s_fp, s_fn)
    print(f"\n[지표 2] selection+label  (rationale 문장까지, 더 엄격)")
    print(f"  sentence P={sp:.3f} R={sr:.3f} F1={sf:.3f}  (tp={s_tp} fp={s_fp} fn={s_fn})")

    _print_confusion(_confusion(label_pairs))
    print()


if __name__ == "__main__":
    asyncio.run(main())
