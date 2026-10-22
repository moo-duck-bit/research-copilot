"""테스트용 가짜 LLM·임베딩과 프로젝트 준비 도우미 (실제 API 호출 없음)."""
from __future__ import annotations

import hashlib
import math
import uuid
from collections.abc import Callable
from typing import Any

from sqlalchemy import text

from app.core.db import SessionLocal
from app.models.project import Project
from app.retrieval.chunking import keyword_set


class FakeEmbedder:
    """키워드를 해시 버킷에 흩뿌린 bag-of-words 벡터. 겹치는 단어가 많을수록 가깝다."""

    def __init__(self, dim: int = 1024) -> None:
        self._dim = dim
        self.calls: list[tuple[str, int]] = []

    @property
    def dim(self) -> int:
        return self._dim

    def _vec(self, t: str) -> list[float]:
        v = [0.0] * self._dim
        for k in keyword_set(t) or {t}:
            i = int(hashlib.md5(k.encode()).hexdigest(), 16) % self._dim
            v[i] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    async def embed(self, texts: list[str], *, input_type: str = "document") -> list[list[float]]:
        self.calls.append((input_type, len(texts)))
        return [self._vec(t) for t in texts]


class FakeLLM:
    """노드 가설 문장을 보고 판정을 돌려준다. decide(hypothesis, user_prompt) -> dict | Exception."""

    model = "fake-model"

    def __init__(self, decide: Callable[[str, str], Any]) -> None:
        self.decide = decide
        self.prompts: list[tuple[str, str]] = []  # (hypothesis, user prompt)
        self.budget_needed: list[int] = []
        self.budget_error: Exception | None = None

    async def ensure_budget(self, needed: int) -> None:
        self.budget_needed.append(needed)
        if self.budget_error:
            raise self.budget_error

    async def extract(self, *, system, user, tools, tool_choice=None):
        hyp = user.split("<가설>\n", 1)[1].split("\n</가설>", 1)[0]
        self.prompts.append((hyp, user))
        out = self.decide(hyp, user)
        if isinstance(out, BaseException):
            raise out
        return tools[0]["name"], out


def judgment(stance: str, quotes: list[str] | None = None, **factors) -> dict:
    return {
        "stance": stance,
        "rationale_sentences": quotes or [],
        "warrant": "테스트 판정",
        "provided_evidence_types": ["financial_performance"] if quotes else [],
        "factors": {"evidence_type_fit": 0.8, "source_credibility": 0.7, **factors},
    }


def node(nid: str, hyp: str, parent: str | None = None, desc: str = "", level: int = 1) -> dict:
    return {
        "node_id": nid,
        "parent_id": parent,
        "level": level,
        "hypothesis": hyp,
        "sub_questions": [hyp + "?"],
        "required_evidence": [
            {
                "evidence_type": "financial_performance",
                "specificity": "quantitative",
                "recency_required_months": 12,
                "description": desc or hyp,
            }
        ],
        "priority": 3,
        "is_leaf": True,
        "mece_rationale": "테스트",
    }


async def create_project(nodes: list[dict], question: str = "테스트 핵심 질문") -> str:
    async with SessionLocal() as s:
        p = Project(
            brief="테스트",
            status="complete",
            primary_question=question,
            tree={"primary_question": question, "hypothesis_tree": nodes},
        )
        s.add(p)
        await s.commit()
        return str(p.id)


async def drop_project(pid: str) -> None:
    async with SessionLocal() as s:
        for t in ("content_items", "node_embeddings"):
            await s.execute(text(f"DELETE FROM {t} WHERE project_id=:p"), {"p": pid})
        await s.execute(text("DELETE FROM projects WHERE id=:p"), {"p": pid})
        await s.commit()


def new_uuid() -> str:
    return str(uuid.uuid4())
