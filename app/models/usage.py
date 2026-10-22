"""LLM 토큰 사용량 원장. 일일 예산을 모든 파드가 함께 쓰도록 DB에 둔다."""
from __future__ import annotations

import datetime

from sqlalchemy import BigInteger, Date, Integer
from sqlalchemy.orm import Mapped, mapped_column

from app.models.project import Base


class LlmUsage(Base):
    __tablename__ = "llm_usage"

    day: Mapped[datetime.date] = mapped_column(Date, primary_key=True)
    input_tokens: Mapped[int] = mapped_column(BigInteger, default=0)
    output_tokens: Mapped[int] = mapped_column(BigInteger, default=0)
    calls: Mapped[int] = mapped_column(Integer, default=0)
