"""웹 요청과 분리해 돌리는 분석 작업 관리.

분석은 노드 수만큼 LLM을 불러 수십 초~수 분이 걸린다. 요청 하나로 끝까지 붙잡으면
프록시 제한 시간(300초)에 끊기거나, 사용자가 기다리다 새로고침해서 결과를 잃는다.
그래서 웹은 접수만 받고 바로 돌려준 뒤, 작업은 여기서 돌리고 결과는 DB에 남긴다.

파드가 종료 신호를 받으면 진행 중인 작업이 끝나기를 일정 시간 기다린다(drain).
그 안에 못 끝난 작업은 '처리 중'으로 남고, 일정 시간이 지나면 다시 돌릴 수 있게 된다.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

log = logging.getLogger("app.tasks")
_running: set[asyncio.Task] = set()


def spawn(coro: Coroutine[Any, Any, Any], name: str | None = None) -> asyncio.Task:
    task = asyncio.create_task(coro, name=name)
    _running.add(task)  # 참조를 쥐고 있어야 작업이 중간에 GC되지 않는다.

    def _done(t: asyncio.Task) -> None:
        _running.discard(t)
        if not t.cancelled() and t.exception() is not None:
            log.warning("백그라운드 작업 실패 %s: %s", t.get_name(), t.exception())

    task.add_done_callback(_done)
    return task


def running_count() -> int:
    return len(_running)


async def drain(timeout: float) -> None:
    pending = [t for t in _running if not t.done()]
    if not pending:
        return
    log.info("종료 전 진행 중인 분석 %d건을 최대 %.0f초 기다린다", len(pending), timeout)
    _done, still = await asyncio.wait(pending, timeout=timeout)
    for t in still:
        t.cancel()
    if still:  # 취소된 작업이 '끊김' 상태를 DB에 남길 짧은 시간을 준다.
        await asyncio.wait(still, timeout=5)
