"""1단계 후보 필터: 자료 청킹 + 노드 임베딩(pgvector) top-k + 키워드 보완.

임베딩 유사도는 오직 '후보를 좁히는' 용도로만 쓴다. 실제 관련성 판정은
2~4단계(스탠스/rationale/warrant/강도)에서 이루어진다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import text

from app.core.db import SessionLocal
from app.llm.embeddings import EmbeddingProvider
from app.retrieval.chunking import Chunk, chunk_text, keyword_scores  # noqa: F401 (재노출)

# 키워드로만 걸린 노드에 내줄 자리 수와 기준 점수.
KEYWORD_SLOTS = 2
KEYWORD_MIN_SCORE = 0.25


def node_embed_text(node: dict, nodes: list[dict] | None = None) -> str:
    """노드 임베딩용 텍스트: 상위 가설 경로 + 가설 + 하위 질문 + required_evidence.

    가설 한 줄만 넣으면 형제 노드끼리 벡터가 거의 같아진다. 상위 가설과 하위 질문을 붙여
    어느 가지의 무슨 질문인지가 벡터에 드러나게 한다.
    """
    parts: list[str] = []
    if nodes:
        by_id = {n["node_id"]: n for n in nodes}
        path: list[str] = []
        pid = node.get("parent_id")
        seen: set[str] = set()
        while pid and pid in by_id and pid not in seen:
            seen.add(pid)
            path.append(by_id[pid].get("hypothesis", ""))
            pid = by_id[pid].get("parent_id")
        if path:
            parts.append("상위: " + " > ".join(reversed(path)))
    parts.append(node.get("hypothesis", ""))
    parts.extend(node.get("sub_questions", []) or [])
    for e in node.get("required_evidence", []):
        if e.get("description"):
            parts.append(e["description"])
    return " / ".join(p for p in parts if p)


def _vec_literal(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in v) + "]"


async def ensure_node_embeddings(
    project_id: str, nodes: list[dict], provider: EmbeddingProvider
) -> None:
    """프로젝트 노드 임베딩을 pgvector에 준비한다.

    없는 노드뿐 아니라 임베딩 텍스트가 바뀐 노드도 다시 계산한다(노드 내용이나 임베딩
    텍스트 규칙이 바뀌었는데 예전 벡터로 후보를 뽑는 일을 막는다).
    테이블과 HNSW 인덱스는 앱 시작 시 init_models()가 만든다.
    """
    wanted = {n["node_id"]: node_embed_text(n, nodes) for n in nodes}
    async with SessionLocal() as session:
        rows = (
            await session.execute(
                text("SELECT node_id, node_text FROM node_embeddings WHERE project_id=:pid"),
                {"pid": project_id},
            )
        ).all()
        have = {nid: t for nid, t in rows}
        stale = [nid for nid, t in wanted.items() if have.get(nid) != t]
        if not stale:
            return

        embeddings = await provider.embed([wanted[n] for n in stale], input_type="document")
        for nid, emb in zip(stale, embeddings):
            await session.execute(
                text(
                    "INSERT INTO node_embeddings (project_id, node_id, node_text, embedding)"
                    " VALUES (:pid, :nid, :txt, CAST(:emb AS vector))"
                    " ON CONFLICT (project_id, node_id) DO UPDATE SET"
                    " node_text=EXCLUDED.node_text, embedding=EXCLUDED.embedding"
                ),
                {"pid": project_id, "nid": nid, "txt": wanted[nid], "emb": _vec_literal(emb)},
            )
        await session.commit()


@dataclass
class Candidates:
    ranked: list[tuple[str, float]]  # (node_id, 최고 코사인 유사도; 키워드로만 걸리면 0.0)
    chunks_by_node: dict[str, list[int]] = field(default_factory=dict)  # 관련 청크, 중요한 순
    via: dict[str, str] = field(default_factory=dict)  # 'embedding' | 'keyword' | 'both'


async def retrieve_candidate_detail(
    project_id: str,
    chunks: list[Chunk],
    nodes: list[dict],
    provider: EmbeddingProvider,
    *,
    per_chunk_k: int = 3,
    max_candidates: int = 5,
) -> Candidates:
    """각 청크의 임베딩 top-k 노드를 모으고, 키워드로만 걸린 노드에 자리를 일부 내준다."""
    chunk_embs = await provider.embed([c.text for c in chunks], input_type="query")
    # 짧은 자료는 청크가 1~2개뿐이라 청크당 3개로는 후보 상한을 못 채운다.
    # 청크 수가 적으면 청크당 더 많이 뽑아 후보 상한까지 볼 수 있게 한다.
    per_chunk_k = min(max(per_chunk_k, -(-max_candidates // max(1, len(chunks)))), max_candidates)
    best: dict[str, float] = {}
    hits: dict[str, list[tuple[float, int]]] = {}
    async with SessionLocal() as session:
        # HNSW 인덱스는 근사 이웃을 먼저 뽑고 WHERE(project_id)를 나중에 거른다.
        # 프로젝트가 많아지면 걸러진 뒤 k개가 안 남을 수 있어, pgvector의 반복 스캔을
        # 켜서 k개가 찰 때까지 인덱스를 더 읽게 한다(pgvector 0.8+).
        await session.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))
        for ci, emb in enumerate(chunk_embs):
            rows = (
                await session.execute(
                    text(
                        "SELECT node_id, 1 - (embedding <=> CAST(:q AS vector)) AS sim"
                        " FROM node_embeddings WHERE project_id=:pid"
                        " ORDER BY embedding <=> CAST(:q AS vector) ASC LIMIT :k"
                    ),
                    {"q": _vec_literal(emb), "pid": project_id, "k": per_chunk_k},
                )
            ).all()
            for node_id, sim in rows:
                sim = float(sim)
                hits.setdefault(node_id, []).append((sim, ci))
                if node_id not in best or sim > best[node_id]:
                    best[node_id] = sim

    dense = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    picked = dense[:max_candidates]
    picked_ids = {nid for nid, _ in picked}
    via = {nid: "embedding" for nid in picked_ids}

    # 키워드 보완: 임베딩 후보에 못 든 노드 중 약어·숫자·고유명사가 겹치는 노드
    kw = keyword_scores({n["node_id"]: node_embed_text(n, nodes) for n in nodes}, chunks)
    extras = sorted(
        (
            (nid, score, ci)
            for nid, (score, ci) in kw.items()
            if nid not in picked_ids and score >= KEYWORD_MIN_SCORE and ci >= 0
        ),
        key=lambda t: t[1],
        reverse=True,
    )[:KEYWORD_SLOTS]
    if extras:
        keep = max(0, max_candidates - len(extras))
        picked = picked[:keep]
        picked_ids = {nid for nid, _ in picked}
        via = {nid: "embedding" for nid in picked_ids}
        for nid, _score, ci in extras:
            picked.append((nid, best.get(nid, 0.0)))
            via[nid] = "keyword"
            hits.setdefault(nid, []).append((1.0, ci))
    for nid, _ in picked:
        score, ci = kw.get(nid, (0.0, -1))
        if via[nid] == "embedding" and score >= KEYWORD_MIN_SCORE:
            via[nid] = "both"
            hits.setdefault(nid, []).append((score, ci))

    chunks_by_node: dict[str, list[int]] = {}
    for nid, _ in picked:
        order: list[int] = []
        for _s, ci in sorted(hits.get(nid, []), reverse=True):
            if ci not in order:
                order.append(ci)
        chunks_by_node[nid] = order
    return Candidates(ranked=picked, chunks_by_node=chunks_by_node, via=via)


async def retrieve_candidates(
    project_id: str,
    chunks: list[str],
    provider: EmbeddingProvider,
    *,
    per_chunk_k: int = 3,
    max_candidates: int = 5,
) -> list[tuple[str, float]]:
    """이전 호출부 호환용: 임베딩 top-k만으로 (node_id, sim) 목록을 돌려준다."""
    objs = [Chunk(0, 0, c) for c in chunks]
    res = await retrieve_candidate_detail(
        project_id, objs, [], provider, per_chunk_k=per_chunk_k, max_candidates=max_candidates
    )
    return res.ranked
