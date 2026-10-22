"""프로젝트(리서치 목표) DB 모델. 가설 트리 JSON을 저장한다.

3단계에서 노드 임베딩(pgvector)을 추가할 예정. 지금은 트리 JSON 저장까지만.
"""
from __future__ import annotations

import datetime
import uuid

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    brief: Mapped[str] = mapped_column(Text)
    # 'complete' | 'needs_clarification'
    status: Mapped[str] = mapped_column(String(32))
    primary_question: Mapped[str | None] = mapped_column(Text, nullable=True)
    # HypothesisTree.model_dump(mode="json") 전체를 저장
    tree: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    clarification_question: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
