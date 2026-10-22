"""tool-use 입력의 '이중 직렬화' 보정.

모델이 배열·객체 필드를 실제 배열·객체 대신 JSON 문자열('[...]', '{...}')로 넣는 경우가 있다.
그대로 검증하면 'list 기대, str 받음'으로 실패하고, 같은 프롬프트로 재시도해도 같은 실수를
반복하기 쉽다. 검증 전에 해당 필드만 한 번 풀어 준다. 풀리지 않으면 원래 값을 두어
평소처럼 검증 오류가 나게 한다.
"""
from __future__ import annotations

import json
from typing import Any


def loads_json_fields(data: Any, fields: tuple[str, ...]) -> Any:
    if not isinstance(data, dict):
        return data
    out = dict(data)
    for f in fields:
        v = out.get(f)
        if isinstance(v, str) and v.strip()[:1] in ("[", "{"):
            try:
                out[f] = json.loads(v)
            except ValueError:
                pass
    return out
