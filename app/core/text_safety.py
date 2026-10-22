"""붙여넣은 텍스트를 저장·분석하기 전에 정리하고 검사한다.

웹·PDF·HWP·메신저에서 복사한 텍스트에는 눈에 안 보이는 문자가 섞여 들어온다.
- NUL(\\x00): PostgreSQL text에 저장할 수 없어, LLM 비용을 다 쓴 뒤 저장 단계에서 실패한다.
- 짝 없는 서로게이트: 이모지가 잘린 흔적. 상위 API가 400으로 거부하고 재시도해도 같다.
- 제로폭·방향 제어 문자: 화면엔 안 보이지만 지시문을 숨기는 데 쓰이고, 근거 대조를 깨뜨린다.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

# 줄바꿈(\n)과 탭(\t)을 뺀 C0 제어문자 + DEL
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SURROGATE = re.compile(r"[\ud800-\udfff]")
# 제로폭 공백·결합자, 방향 제어(LRE/RLO 등), 단어 결합자, BOM
_INVISIBLE = re.compile(r"[​-‏‪-‮⁠-⁤⁦-⁩﻿­]")
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# 프롬프트에서 자료를 감싸는 데 쓰는 태그 이름. 자료 안에 같은 태그가 있으면 경계가 깨진다.
PROMPT_TAGS = ("자료", "가설", "이 가설의 하위 질문", "이 가설이 요구하는 근거(required_evidence)")
_TAG_RE = re.compile(
    r"<\s*(/?)\s*(" + "|".join(re.escape(t) for t in PROMPT_TAGS) + r")\s*>"
)


def sanitize_text(s: str) -> str:
    """저장·분석에 쓸 텍스트로 정리한다. 글자 내용은 바꾸지 않고 '보이지 않는 것'만 지운다."""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _SURROGATE.sub("", s)
    s = _CONTROL.sub("", s)
    s = _INVISIBLE.sub("", s)
    # 한글 자모가 분리된 NFD(맥 파일명·일부 PDF)를 완성형으로 맞춘다.
    s = unicodedata.normalize("NFC", s)
    return s.strip()


def is_url_only(s: str) -> bool:
    """본문 없이 URL만 붙여넣었는지. 서버는 URL 내용을 가져오지 않는다."""
    if not _URL.search(s):
        return False
    rest = _URL.sub("", s)
    return len(re.sub(r"\s+", "", rest)) < 20


def content_hash(s: str) -> str:
    """중복 제출 판별용 해시. 공백 차이(줄바꿈·연속 공백)는 같은 자료로 본다."""
    canon = re.sub(r"\s+", " ", s).strip()
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def neutralize_prompt_tags(s: str) -> str:
    """자료 속 '</자료>' 같은 태그를 전각 꺾쇠로 바꿔 프롬프트 경계를 지킨다.

    전각 꺾쇠는 NFKC 정규화에서 원래 꺾쇠로 돌아오므로 근거 문장 원문 대조에는 영향이 없다.
    """
    return _TAG_RE.sub(lambda m: f"＜{m.group(1)}{m.group(2)}＞", s)
