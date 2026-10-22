"""MCP 툴 5개가 FastMCP 서버에 등록됐는지 확인."""
import asyncio

from app.mcp.server import mcp

EXPECTED_TOOLS = {
    "set_project_brief",
    "analyze_content",
    "find_synthesis_candidates",
    "check_goal_drift",
    "list_collected",
}


def test_five_tools_registered():
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    assert names == EXPECTED_TOOLS
