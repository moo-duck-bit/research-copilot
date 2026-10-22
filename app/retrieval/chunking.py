"""자료 청킹, 노드별 발췌 만들기, 키워드 겹침 점수.

청크는 원문 위치(start, end)로 들고 다닌다. 그래야 판정 단계에서 '관련 청크와 그 앞뒤'만
원문 그대로 잘라 프롬프트에 넣을 수 있다.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

# 문장 끝(마침표류 + 공백) 또는 줄바꿈. "3.5%"처럼 뒤에 공백이 없는 마침표는 끊지 않는다.
_BOUNDARY = re.compile(r"(?<=[.!?。])\s+|\n+")
OMIT_MARK = "\n[… 중략 …]\n"


@dataclass(frozen=True)
class Chunk:
    start: int
    end: int
    text: str  # 공백을 접은 임베딩용 텍스트


def _sentence_spans(s: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in _BOUNDARY.finditer(s):
        spans.append((pos, m.start()))
        pos = m.end()
    spans.append((pos, len(s)))
    out = []
    for a, b in spans:
        while a < b and s[a].isspace():
            a += 1
        while b > a and s[b - 1].isspace():
            b -= 1
        if b > a:
            out.append((a, b))
    return out


def _hard_split(s: str, a: int, b: int, max_chars: int, overlap: int) -> list[tuple[int, int]]:
    """마침표 없는 긴 구간(표·개조식·PDF 복사)을 겹치는 창으로 자른다.

    창 끝은 가능하면 뒤쪽 30% 안의 공백·쉼표·구분자에 맞춰 단어 중간을 피한다.
    """
    out: list[tuple[int, int]] = []
    cur = a
    while cur < b:
        end = min(cur + max_chars, b)
        if end < b:
            floor = cur + int(max_chars * 0.7)
            cut = max(s.rfind(ch, floor, end) for ch in (" ", ",", "|", "·", ";", "\t"))
            if cut > floor:
                end = cut + 1
        out.append((cur, end))
        if end >= b:
            break
        cur = max(end - overlap, cur + 1)
    return out


def chunk_spans(s: str, max_chars: int = 400, overlap: int = 60) -> list[Chunk]:
    pieces: list[tuple[int, int]] = []
    for a, b in _sentence_spans(s):
        if b - a > max_chars:
            pieces.extend(_hard_split(s, a, b, max_chars, overlap))
        else:
            pieces.append((a, b))

    chunks: list[Chunk] = []
    cur: tuple[int, int] | None = None
    for a, b in pieces:
        if cur is None:
            cur = (a, b)
        elif b - cur[0] <= max_chars and a >= cur[1]:
            cur = (cur[0], b)
        else:
            chunks.append(_mk(s, *cur))
            cur = (a, b)
    if cur is not None:
        chunks.append(_mk(s, *cur))
    if not chunks and s.strip():
        chunks.append(_mk(s, 0, len(s)))
    return chunks


def _mk(s: str, a: int, b: int) -> Chunk:
    return Chunk(a, b, re.sub(r"\s+", " ", s[a:b]).strip())


def chunk_text(s: str, max_chars: int = 400) -> list[str]:
    return [c.text for c in chunk_spans(s, max_chars)]


def build_excerpt(
    s: str,
    chunks: list[Chunk],
    focus: list[int],
    *,
    neighbors: int = 1,
    max_chars: int = 6000,
) -> str:
    """focus 청크와 그 앞뒤 neighbors개만 원문 그대로 이어 붙인다. 빠진 곳은 중략 표시.

    focus는 중요한 순서대로 받는다. 상한에 닿으면 덜 중요한 청크부터 뺀다.
    """
    picked: set[int] = set()
    total = 0
    for i in focus:
        group = [j for j in range(i - neighbors, i + neighbors + 1) if 0 <= j < len(chunks)]
        add = [j for j in group if j not in picked]
        size = sum(chunks[j].end - chunks[j].start for j in add)
        if picked and total + size > max_chars:
            continue
        picked.update(add)
        total += size
    if not picked:
        return ""

    order = sorted(picked)
    parts: list[str] = []
    prev_end = None
    for j in order:
        a, b = chunks[j].start, chunks[j].end
        if prev_end is not None and a < prev_end:
            a = prev_end  # 겹치는 창은 이어 붙인다
        if prev_end is None:
            if a > 0:
                parts.append(OMIT_MARK.lstrip("\n"))
        elif s[prev_end:a].strip():
            parts.append(OMIT_MARK)
        else:
            parts.append(s[prev_end:a])
        parts.append(s[a:b])
        prev_end = max(b, prev_end or 0)
    if prev_end is not None and s[prev_end:].strip():
        parts.append(OMIT_MARK.rstrip("\n"))
    return "".join(parts).strip()


# --- 키워드 겹침 (임베딩이 약한 약어·숫자·고유명사 보완) ---
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9&\-]+|\d+(?:[.,]\d+)?%?|[가-힣]+")


def keyword_set(s: str) -> set[str]:
    """영문 단어(소문자), 숫자(단위 포함), 한글은 2글자 조각으로 쪼갠 집합.

    한글은 조사·어미가 붙어 단어 단위로는 잘 안 맞으므로 2글자 조각으로 비교한다.
    """
    out: set[str] = set()
    for t in _TOKEN.findall(s):
        if t[0].isdigit():
            if len(t) >= 2 or t.endswith("%"):
                out.add(t.replace(",", ""))
        elif "가" <= t[0] <= "힣":
            out.update(t[i : i + 2] for i in range(len(t) - 1))
        elif len(t) >= 2:
            out.add(t.lower())
    return out


def keyword_scores(node_texts: dict[str, str], chunks: list[Chunk]) -> dict[str, tuple[float, int]]:
    """노드마다 (가장 잘 겹친 청크의 점수, 그 청크 번호). 점수는 노드 키워드의 IDF 가중 비율.

    여러 노드에 공통인 단어(예: 시장, 매출)는 가중치가 낮아 형제 노드를 구분하는 단어가 더 크게 친다.
    """
    sets = {nid: keyword_set(t) for nid, t in node_texts.items()}
    n = len(sets)
    df: dict[str, int] = {}
    for ks in sets.values():
        for k in ks:
            df[k] = df.get(k, 0) + 1
    # 영문 약어·숫자는 한글 2글자 조각보다 우연히 겹칠 확률이 낮아 두 배로 친다.
    idf = {
        k: math.log((n + 1) / (c + 0.5)) * (1.0 if "가" <= k[0] <= "힣" else 2.0)
        for k, c in df.items()
    }

    chunk_sets = [keyword_set(c.text) for c in chunks]
    out: dict[str, tuple[float, int]] = {}
    for nid, ks in sets.items():
        denom = sum(idf[k] for k in ks) or 1.0
        best = (0.0, -1)
        for ci, cs in enumerate(chunk_sets):
            score = sum(idf[k] for k in ks & cs) / denom
            if score > best[0]:
                best = (score, ci)
        out[nid] = best
    return out
