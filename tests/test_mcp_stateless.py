"""MCP 세션 404 재현과 무상태 전환 확인.

파드 두 개를 같은 FastMCP 서버로 만든 두 ASGI 앱으로 흉내 낸다.
- 상태 유지 모드: 파드 A에서 만든 세션으로 파드 B에 요청하면 404 (슬라이드의 장애)
- 무상태 모드: 어느 파드로 가도 처리된다 (조치)
"""
import uuid

from fastapi.testclient import TestClient

from app.main import app
from app.mcp.server import mcp

HEADERS = {"accept": "application/json, text/event-stream", "content-type": "application/json"}
INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"},
    },
}
LIST = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
INITIALIZED = {"jsonrpc": "2.0", "method": "notifications/initialized"}


def _pod(stateless: bool):
    return mcp.http_app(path="/mcp", stateless_http=stateless, json_response=True)


def test_stateful_session_is_lost_on_other_pod():
    with TestClient(_pod(False)) as pod_a, TestClient(_pod(False)) as pod_b:
        r = pod_a.post("/mcp", json=INIT, headers=HEADERS)
        assert r.status_code == 200
        sid = r.headers["mcp-session-id"]
        h = {**HEADERS, "mcp-session-id": sid}
        pod_a.post("/mcp", json=INITIALIZED, headers=h)

        assert pod_a.post("/mcp", json=LIST, headers=h).status_code == 200  # 같은 파드는 됨
        assert pod_b.post("/mcp", json=LIST, headers=h).status_code == 404  # 다른 파드는 세션 유실


def test_stateless_any_pod_handles_request():
    with TestClient(_pod(True)) as pod_a, TestClient(_pod(True)) as pod_b:
        assert pod_a.post("/mcp", json=INIT, headers=HEADERS).status_code == 200
        r = pod_b.post("/mcp", json=LIST, headers=HEADERS)  # 초기화 안 한 다른 파드
        assert r.status_code == 200
        assert len(r.json()["result"]["tools"]) == 5


def test_app_mounts_mcp_in_stateless_mode():
    # 실제 앱(/mcp-server/mcp): 처음 보는 세션 ID로 와도 404가 아니다.
    with TestClient(app) as client:
        h = {**HEADERS, "mcp-session-id": uuid.uuid4().hex}
        r = client.post("/mcp-server/mcp", json=LIST, headers=h)
        assert r.status_code == 200
        names = {t["name"] for t in r.json()["result"]["tools"]}
        assert names == {
            "set_project_brief",
            "analyze_content",
            "list_collected",
            "find_synthesis_candidates",
            "check_goal_drift",
        }


def _call(client, name, args):
    body = {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": name, "arguments": args}}
    r = client.post("/mcp-server/mcp", json=body, headers=HEADERS)
    assert r.status_code == 200
    return r.json()["result"]


def test_mcp_tool_errors_carry_cause_code():
    # 호스트 모델이 UUID를 지어내거나 한 글자 틀려도 500 대신 원인 코드가 돌아간다.
    with TestClient(app) as client:
        res = _call(client, "list_collected", {"project_id": "proj-123"})
        assert res["isError"] is True
        assert "[invalid_project_id]" in res["content"][0]["text"]


def test_mcp_tool_result_keeps_korean_literal():
    # 결과 텍스트에 한글이 \\uXXXX로 이스케이프되면 호스트 모델이 직접 디코딩하다 틀린다.
    from tests.fakes import create_project, drop_project, node

    with TestClient(app) as client:
        pid = client.portal.call(create_project, [node("1", "가설 하나"), node("2", "가설 둘")], "한글 핵심 질문")
        try:
            res = _call(client, "list_collected", {"project_id": pid})
            assert res["isError"] is False
            text = res["content"][0]["text"]
            assert "한글 핵심 질문" in text and "\\u" not in text
        finally:
            client.portal.call(drop_project, pid)
