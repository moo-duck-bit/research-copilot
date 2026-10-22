"""근거 문장 원문 대조 (환각 방지).

모델이 '그대로 발췌'하라는 지시를 받아도 따옴표 종류, 띄어쓰기, 전각 문자, 줄바꿈을
바꿔 인용하는 일이 잦다. 공백만 정규화하던 이전 검사는 이런 인용을 '원문에 없음'으로
버렸고, 그 결과 지지/반증이 아무 표시 없이 약한 관련으로 강등됐다.

여기서는 단계적으로 대조한다.
1) 표기 정규화(NFKC, 따옴표·대시·말줄임 통일, 공백 접기) 후 부분 문자열
2) 띄어쓰기를 무시한 부분 문자열 (글자와 순서는 그대로여야 한다)
3) 그래도 없으면 근사 일치(글자 92% 이상 같고 길이가 비슷한 원문 구간)

어느 단계로 찾든, 화면에 보여 주는 문장은 모델이 쓴 문장이 아니라 원문에서 잘라낸 구간이다.
너무 짧은 인용(한두 단어)은 어떤 주장에도 갖다 붙일 수 있어 근거로 인정하지 않는다.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

MIN_QUOTE_CHARS = 6  # 띄어쓰기 제외 글자 수
FUZZY_MIN_CHARS = 15
FUZZY_RATIO = 0.92

_QUOTES = str.maketrans(
    {
        "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"', "«": '"', "»": '"',
        "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'", "`": "'",
        "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-", "−": "-",
        "…": "...", "·": "·", "ㆍ": "·", "•": "·",
    }
)
_INVISIBLE = re.compile(r"[​-‏‪-‮⁠-⁤﻿­]")
# PDF 줄끝 하이픈: "inter-\nnational" → "international"
_HYPHEN_BREAK = re.compile(r"(?<=[A-Za-z])-\n(?=[a-z])")
_EDGE = " \t\n\"'.,;:"


def _canon_char(ch: str) -> str:
    return unicodedata.normalize("NFKC", ch).translate(_QUOTES)


def normalize(s: str) -> str:
    s = _HYPHEN_BREAK.sub("", s)
    s = _INVISIBLE.sub("", s)
    s = unicodedata.normalize("NFKC", s).translate(_QUOTES)
    return re.sub(r"\s+", " ", s).strip()


def _nospace_index(source: str) -> tuple[str, list[int]]:
    """원문을 정규화하고 공백을 뺀 문자열과, 각 글자의 원문 위치를 함께 만든다."""
    chars: list[str] = []
    pos: list[int] = []
    text = _HYPHEN_BREAK.sub(lambda m: "\0\0", source)  # 길이 유지(위치 보존)
    for i, ch in enumerate(text):
        if ch == "\0" or ch.isspace() or _INVISIBLE.match(ch):
            continue
        for c in _canon_char(ch):
            if c.isspace():
                continue
            chars.append(c)
            pos.append(i)
    return "".join(chars), pos


def _nospace(s: str) -> str:
    return re.sub(r"\s+", "", normalize(s))


@dataclass
class VerbatimResult:
    kept: list[str] = field(default_factory=list)  # 원문에서 잘라낸 그대로의 문장
    offsets: list[tuple[int, int]] = field(default_factory=list)  # 원문 [start, end)
    dropped: int = 0  # 원문에서 못 찾았거나 너무 짧아 버린 수
    repaired: int = 0  # 표기 차이를 보정해 원문 구간으로 바꾼 수


def verify_rationale(sentences: list[str], source: str) -> VerbatimResult:
    out = VerbatimResult()
    src_ns, pos = _nospace_index(source)
    seen: set[tuple[int, int]] = set()

    for raw in sentences:
        q_full = _nospace(raw)
        q = q_full.strip(_EDGE)  # 앞뒤 따옴표·마침표는 모델이 붙이거나 뺄 수 있다
        if len(q) < MIN_QUOTE_CHARS:
            out.dropped += 1
            continue

        how = "exact"
        start, length = -1, 0
        for cand in dict.fromkeys((q_full, q)):  # 마침표까지 맞으면 그대로, 아니면 떼고
            start = src_ns.find(cand)
            if start >= 0:
                length = len(cand)
                break
        if start < 0:
            start, length = _fuzzy_find(src_ns, q)
            how = "fuzzy" if start >= 0 else "none"

        if start < 0:
            out.dropped += 1
            continue

        a = pos[start]
        b = pos[start + length - 1] + 1
        span = (a, b)
        if span in seen:  # 같은 문장을 두 번 인용
            continue
        seen.add(span)
        quote = source[a:b].strip()
        out.kept.append(quote)
        out.offsets.append(span)
        if how != "exact" or normalize(raw).strip(_EDGE) != normalize(quote).strip(_EDGE):
            out.repaired += 1
    return out


def _fuzzy_find(src: str, q: str) -> tuple[int, int]:
    """q와 가장 비슷한 원문 구간을 찾는다. 기준 미달이면 (-1, 0)."""
    if len(q) < FUZZY_MIN_CHARS or not src:
        return -1, 0
    sm = SequenceMatcher(None, src, q, autojunk=False)
    m = sm.find_longest_match(0, len(src), 0, len(q))
    if m.size < max(5, len(q) // 4):
        return -1, 0
    # 가장 긴 일치 구간을 기준으로 인용 길이만큼의 원문 창을 맞춰 본다.
    approx = m.a - m.b
    best = (-1, 0, 0.0)
    for delta in range(-3, 4):
        for extra in (-2, -1, 0, 1, 2):
            st = max(0, approx + delta)
            ln = len(q) + extra
            win = src[st : st + ln]
            if not win:
                continue
            r = SequenceMatcher(None, win, q, autojunk=False).ratio()
            if r > best[2]:
                best = (st, len(win), r)
    if best[2] >= FUZZY_RATIO:
        return best[0], best[1]
    return -1, 0
