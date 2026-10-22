"""임베딩 추상화. 후보 필터(1단계) 전용이며, 판단 로직처럼 서버 내부에 캡슐화한다.

Anthropic은 임베딩 API를 제공하지 않으므로 Voyage AI를 구현체로 둔다.
다른 임베딩 모델로 교체하더라도 이 Protocol 뒤에서만 바뀐다.

- 요청 하나에 넣는 텍스트 수와 글자 수에 상한을 두고 나눠 보낸다(요청당 토큰 한도 초과 방지).
- 클라이언트는 프로세스에서 하나만 만들어 재사용한다.
- 429 대기 시간의 총합에 상한을 둔다. 예전에는 25초씩 최대 5번(약 2분)을 기다려
  뒤따르는 LLM 호출과 합쳐 프록시 300초 제한을 넘길 수 있었다.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

from app.core.config import settings
from app.llm.base import EmbeddingUnavailableError, LLMConfigError


@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def dim(self) -> int:
        ...

    async def embed(
        self, texts: list[str], *, input_type: str = "document"
    ) -> list[list[float]]:
        """input_type: 'document'(색인 대상) 또는 'query'(검색어)."""
        ...


_clients: dict[str, Any] = {}


def _shared_client(api_key: str) -> Any:
    import voyageai

    if api_key not in _clients:
        _clients[api_key] = voyageai.AsyncClient(api_key=api_key, max_retries=0)
    return _clients[api_key]


def batches(texts: list[str], max_items: int, max_chars: int) -> list[list[str]]:
    out: list[list[str]] = []
    cur: list[str] = []
    size = 0
    for t in texts:
        if cur and (len(cur) >= max_items or size + len(t) > max_chars):
            out.append(cur)
            cur, size = [], 0
        cur.append(t)
        size += len(t)
    if cur:
        out.append(cur)
    return out


def _is_rate_limit(exc: BaseException) -> bool:
    try:
        from voyageai.error import RateLimitError
    except Exception:  # noqa: BLE001
        RateLimitError = ()  # type: ignore[assignment]
    return isinstance(exc, RateLimitError) or getattr(exc, "http_status", None) == 429


class VoyageEmbeddingProvider:
    """Voyage AI 임베딩 구현체. 키는 환경변수(VOYAGE_API_KEY)로만 받는다."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dim: int | None = None,
        *,
        client: Any | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._api_key = api_key or settings.voyage_api_key
        self._model = model or settings.embedding_model
        self._dim = dim or settings.embedding_dim
        self._client = client
        self._sleep = sleep

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(
        self, texts: list[str], *, input_type: str = "document"
    ) -> list[list[float]]:
        if self._client is None and not self._api_key:
            raise LLMConfigError("VOYAGE_API_KEY가 설정되지 않았다. 환경변수로 주입하라.")
        client = self._client or _shared_client(self._api_key)
        out: list[list[float]] = []
        waited = 0.0
        for batch in batches(
            texts, settings.embedding_batch_max_items, settings.embedding_batch_max_chars
        ):
            while True:
                try:
                    resp = await client.embed(batch, model=self._model, input_type=input_type)
                    out.extend(resp.embeddings)
                    break
                except Exception as exc:  # noqa: BLE001
                    if not _is_rate_limit(exc):
                        raise
                    step = settings.embedding_rate_limit_wait_seconds
                    if waited + step > settings.embedding_max_wait_seconds:
                        raise EmbeddingUnavailableError(
                            "임베딩 API 요청 한도에 걸려 제한 시간 안에 처리하지 못했다.",
                            retry_after=step,
                        ) from exc
                    waited += step
                    await self._sleep(step)
        return out
