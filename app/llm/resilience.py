"""LLM 호출 보호: 오류 분류, 재시도 대기 계산, 서킷브레이커.

429(요청 한도)와 529(과부하)를 같은 '실패'로 묶어 무차별 재시도하면 비용과 지연이
같이 불어난다. 오류 코드별로 다르게 다룬다.
"""
from __future__ import annotations

import random
import time
from collections.abc import Callable

import anthropic

from app.llm.base import CircuitOpenError

RATE_LIMIT = "rate_limit"   # 429: 서버가 알려준 만큼 기다렸다 재시도, 브레이커엔 안 셈
TRANSIENT = "transient"     # 529·5xx·408·연결 오류: 짧게 재시도, 브레이커에 셈
FATAL = "fatal"             # 그 밖의 4xx: 요청이 잘못됨, 재시도 안 함

_TRANSIENT_STATUS = {408, 500, 502, 503, 504, 529}


def classify(exc: BaseException) -> str:
    if isinstance(exc, anthropic.APIConnectionError):  # APITimeoutError 포함
        return TRANSIENT
    if isinstance(exc, anthropic.APIStatusError):
        if exc.status_code == 429:
            return RATE_LIMIT
        if exc.status_code in _TRANSIENT_STATUS:
            return TRANSIENT
        return FATAL
    return FATAL


def retry_after_seconds(exc: BaseException) -> float | None:
    """응답 헤더의 retry-after(초)를 읽는다. 없거나 해석 불가면 None."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def backoff_seconds(attempt: int, base: float, rng: Callable[[], float] = random.random) -> float:
    """attempt번째 재시도(1부터)의 대기 시간: base * 2^(attempt-1) + 0~base 지터."""
    return base * (2 ** (attempt - 1)) + base * rng()


class CircuitBreaker:
    """연속 일시 장애가 threshold번 쌓이면 cooldown 동안 호출을 막는다.

    cooldown이 지나면 시험 호출 하나만 통과시킨다(half-open). 성공하면 닫히고,
    실패하면 다시 cooldown만큼 열린다. 상태는 프로세스(파드)마다 따로 가진다.
    """

    def __init__(
        self, threshold: int, cooldown: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.threshold = threshold
        self.cooldown = cooldown
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._trial_in_flight = False

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if self._clock() - self._opened_at < self.cooldown:
            return "open"
        return "half_open"

    def before_call(self) -> None:
        state = self.state
        if state == "open":
            raise CircuitOpenError(self.cooldown - (self._clock() - self._opened_at))
        if state == "half_open":
            if self._trial_in_flight:
                raise CircuitOpenError(1.0)
            self._trial_in_flight = True

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None
        self._trial_in_flight = False

    def record_failure(self) -> None:
        was_trial = self._trial_in_flight
        self._trial_in_flight = False
        self._failures += 1
        if was_trial or self._failures >= self.threshold:
            self._opened_at = self._clock()

    def record_neutral(self) -> None:
        """장애도 성공도 아닌 결과(429·4xx). 시험 호출 자리만 비운다."""
        self._trial_in_flight = False
