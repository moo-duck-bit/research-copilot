"""스키마 준비 후 node_embeddings에 HNSW 인덱스가 있고, 플래너가 그 인덱스를 쓸 수 있는지."""
import asyncio

from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.main import app


async def _inspect():
    eng = create_async_engine(settings.database_url, poolclass=NullPool)
    async with eng.connect() as conn:
        idx = (
            await conn.execute(
                text(
                    "SELECT indexdef FROM pg_indexes WHERE tablename='node_embeddings'"
                    " AND indexname='node_embeddings_embedding_hnsw'"
                )
            )
        ).scalar()
        await conn.execute(text("SET enable_seqscan = off"))
        q = "[" + ",".join(["0"] * (settings.embedding_dim - 1) + ["1"]) + "]"
        plan = (
            await conn.execute(
                text(
                    "EXPLAIN SELECT node_id FROM node_embeddings"
                    " ORDER BY embedding <=> CAST(:q AS vector) LIMIT 3"
                ),
                {"q": q},
            )
        ).scalars().all()
    await eng.dispose()
    return idx, "\n".join(plan)


def test_hnsw_index_exists_and_is_usable():
    with TestClient(app):  # lifespan → init_models()
        pass
    indexdef, plan = asyncio.run(_inspect())
    assert indexdef and "hnsw" in indexdef and "vector_cosine_ops" in indexdef
    assert "node_embeddings_embedding_hnsw" in plan
