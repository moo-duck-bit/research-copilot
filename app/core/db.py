"""DB 엔진 / 세션 / 연결 확인 / 스키마 준비 (PostgreSQL + pgvector)."""
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings

engine = create_async_engine(settings.database_url, pool_pre_ping=True)

SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# 여러 파드가 동시에 떠도 스키마 준비(DDL)는 한 번에 하나만 돌게 하는 advisory lock 키.
_SCHEMA_LOCK_KEY = 7_320_114


async def check_db() -> bool:
    """DB 연결 확인용. 성공하면 True, 실패하면 예외를 던진다."""
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    return True


async def init_models() -> None:
    """스키마를 준비한다(없으면 생성). 이미 있으면 아무것도 바꾸지 않는다.

    - pgvector 확장
    - ORM 테이블: projects, content_items, llm_usage
    - node_embeddings(vector) 테이블과 HNSW 코사인 인덱스
    """
    from app.models import content, usage  # noqa: F401  (테이블 등록)
    from app.models.project import Base

    dim = int(settings.embedding_dim)
    async with engine.begin() as conn:
        await conn.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": _SCHEMA_LOCK_KEY})
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
        # 이전 버전에서 만든 content_items에 새 열을 더한다(기존 행은 done으로 둔다).
        for ddl in (
            "ADD COLUMN IF NOT EXISTS content_hash varchar(64)",
            "ADD COLUMN IF NOT EXISTS status varchar(16) NOT NULL DEFAULT 'done'",
            "ADD COLUMN IF NOT EXISTS error_code varchar(64)",
            "ADD COLUMN IF NOT EXISTS error_message text",
            "ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT now()",
        ):
            await conn.execute(text(f"ALTER TABLE content_items {ddl}"))
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS content_items_project_hash "
                "ON content_items (project_id, content_hash)"
            )
        )
        await conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS node_embeddings ("
                "  project_id uuid NOT NULL, node_id text NOT NULL,"
                f"  node_text text NOT NULL, embedding vector({dim}) NOT NULL,"
                "  PRIMARY KEY (project_id, node_id))"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS node_embeddings_embedding_hnsw "
                "ON node_embeddings USING hnsw (embedding vector_cosine_ops)"
            )
        )
