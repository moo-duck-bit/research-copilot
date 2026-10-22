"""헬스체크가 200을 반환하는지 확인 (DB 연결 포함)."""
from fastapi.testclient import TestClient

from app.main import app


def test_health_ok():
    # with 블록이 lifespan(= FastMCP 세션 매니저 + DB 엔진)을 기동한다.
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["database"] == "connected"
