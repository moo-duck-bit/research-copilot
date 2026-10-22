"""수집 자료 + 분석 결과 저장 모델."""
from __future__ import annotations

import datetime
import uuid

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.project import Base


class ContentItem(Base):
    __tablename__ = "content_items"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    content: Mapped[str] = mapped_column(Text)
    # ContentAnalysis.model_dump(mode="json"). 처리 중이거나 실패한 행은 빈 객체.
    result: Mapped[dict] = mapped_column(JSONB)
    # 같은 프로젝트에 같은 자료가 다시 들어오면 새로 분석하지 않고 이 행을 돌려준다.
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # processing | done | failed. 결과를 읽는 쪽(list_collected 등)은 done만 본다.
    status: Mapped[str] = mapped_column(String(16), server_default="done")
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
