"""liveness / readiness 분리 확인 (DB는 가짜로 대체, 실제 연결 없음).

DB가 죽어도 liveness는 200이어야 한다. 그래야 DB 지연 때 정상 파드까지 재시작되는
크래시 루프가 생기지 않는다. readiness만 503으로 트래픽을 뺀다.
"""
import pytest
from fastapi.testclient import TestClient

import app.main as main


async def _db_down():
    raise ConnectionError("db down")


async def _db_up():
    return True


@pytest.fixture
def client():
    return TestClient(main.app)  # lifespan 없이: DB 초기화를 하지 않는다


def test_liveness_ignores_db(client, monkeypatch):
    monkeypatch.setattr(main, "check_db", _db_down)
    assert client.get("/livez").status_code == 200


def test_readiness_fails_when_db_down(client, monkeypatch):
    monkeypatch.setattr(main, "check_db", _db_down)
    r = client.get("/readyz")
    assert r.status_code == 503
    assert r.json()["database"] == "unavailable"


def test_readiness_ok_when_db_up(client, monkeypatch):
    monkeypatch.setattr(main, "check_db", _db_up)
    assert client.get("/readyz").status_code == 200


async def _db_hangs():
    """멈춘 DB 흉내: 응답이 없고, 취소돼도 연결 정리가 한동안 막힌다."""
    import asyncio

    try:
        await asyncio.sleep(30)
    except asyncio.CancelledError:
        await asyncio.sleep(3)  # 멈춘 소켓을 닫느라 막히는 정리 단계
        raise


def test_readiness_does_not_hang_on_stuck_db(client, monkeypatch):
    import time

    monkeypatch.setattr(main, "check_db", _db_hangs)
    monkeypatch.setattr(main.settings, "readiness_db_timeout_seconds", 0.2)
    t0 = time.monotonic()
    r = client.get("/readyz")
    assert r.status_code == 503
    assert time.monotonic() - t0 < 1.5  # 정리가 막혀도 기다리지 않는다


def test_startup_waits_for_late_db(monkeypatch):
    import asyncio

    calls = {"n": 0}

    async def flaky_init():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("db not up yet")

    async def no_sleep(_):
        return None

    monkeypatch.setattr(main, "init_models", flaky_init)
    monkeypatch.setattr(main.asyncio, "sleep", no_sleep)
    asyncio.run(main._init_models_with_retry(max_wait=60))
    assert calls["n"] == 3  # 두 번 실패 후 세 번째에 성공, 죽지 않음


def test_startup_gives_up_after_max_wait(monkeypatch):
    import asyncio

    async def always_fail():
        raise ConnectionError("db down")

    monkeypatch.setattr(main, "init_models", always_fail)
    with pytest.raises(ConnectionError):
        asyncio.run(main._init_models_with_retry(max_wait=0))
