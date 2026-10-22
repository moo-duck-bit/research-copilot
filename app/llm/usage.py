"""일일 토큰 예산 원장. 모든 파드가 같은 예산을 나눠 쓰도록 DB(llm_usage)에 기록한다."""
from __future__ import annotations

import datetime
from typing import Protocol

from sqlalchemy import text

from app.core.db import SessionLocal


def today(offset_hours: int) -> datetime.date:
    tz = datetime.timezone(datetime.timedelta(hours=offset_hours))
    return datetime.datetime.now(tz).date()


def next_reset(offset_hours: int) -> datetime.datetime:
    """예산이 다시 채워지는 시각(다음 날 0시, 해당 시간대)."""
    tz = datetime.timezone(datetime.timedelta(hours=offset_hours))
    d = datetime.datetime.now(tz).date() + datetime.timedelta(days=1)
    return datetime.datetime(d.year, d.month, d.day, tzinfo=tz)


def estimate_tokens(chars: int) -> int:
    """글자 수로 토큰을 넉넉히 어림한다. 한국어는 영어보다 글자당 토큰이 많아 1글자≈1토큰으로 잡는다."""
    return int(chars)


async def precheck_budget(ledger: "UsageLedger", budget: int, needed: int, offset_hours: int) -> None:
    """호출을 시작하기 전에 이번 작업 전체에 필요한 예산이 남아 있는지 본다.

    노드 5개 중 3개를 판정한 뒤에 예산이 끊기면 앞의 3개 비용만 쓰고 결과는 반쪽이 된다.
    """
    from app.llm.base import BudgetExceededError

    if budget <= 0:
        return
    used = await ledger.used_today()
    if used + needed > budget:
        raise BudgetExceededError(used, budget, needed=needed, resets_at=next_reset(offset_hours))


class UsageLedger(Protocol):
    async def used_today(self) -> int: ...

    async def add(self, input_tokens: int, output_tokens: int) -> None: ...


class DbUsageLedger:
    def __init__(self, offset_hours: int) -> None:
        self.offset_hours = offset_hours

    async def used_today(self) -> int:
        async with SessionLocal() as session:
            row = (
                await session.execute(
                    text("SELECT input_tokens + output_tokens FROM llm_usage WHERE day = :d"),
                    {"d": today(self.offset_hours)},
                )
            ).first()
        return int(row[0]) if row else 0

    async def add(self, input_tokens: int, output_tokens: int) -> None:
        # 여러 파드가 동시에 더해도 덮어쓰지 않도록 DB에서 원자적으로 누적한다.
        async with SessionLocal() as session:
            await session.execute(
                text(
                    "INSERT INTO llm_usage (day, input_tokens, output_tokens, calls)"
                    " VALUES (:d, :i, :o, 1)"
                    " ON CONFLICT (day) DO UPDATE SET"
                    " input_tokens = llm_usage.input_tokens + EXCLUDED.input_tokens,"
                    " output_tokens = llm_usage.output_tokens + EXCLUDED.output_tokens,"
                    " calls = llm_usage.calls + 1"
                ),
                {"d": today(self.offset_hours), "i": input_tokens, "o": output_tokens},
            )
            await session.commit()

    async def snapshot(self) -> dict:
        d = today(self.offset_hours)
        async with SessionLocal() as session:
            row = (
                await session.execute(
                    text("SELECT input_tokens, output_tokens, calls FROM llm_usage WHERE day = :d"),
                    {"d": d},
                )
            ).first()
        i, o, c = (int(row[0]), int(row[1]), int(row[2])) if row else (0, 0, 0)
        return {"day": d.isoformat(), "input_tokens": i, "output_tokens": o, "calls": c}


class InMemoryUsageLedger:
    """테스트용."""

    def __init__(self, used: int = 0) -> None:
        self.used = used
        self.added: list[tuple[int, int]] = []

    async def used_today(self) -> int:
        return self.used

    async def add(self, input_tokens: int, output_tokens: int) -> None:
        self.used += input_tokens + output_tokens
        self.added.append((input_tokens, output_tokens))
