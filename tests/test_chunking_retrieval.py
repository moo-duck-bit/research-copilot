"""청킹·발췌·키워드 보완 (LLM·DB 없음)."""
from app.retrieval.candidates import node_embed_text
from app.retrieval.chunking import OMIT_MARK, build_excerpt, chunk_spans, keyword_scores


def test_unpunctuated_table_is_split_into_bounded_chunks():
    # 마침표 없는 표·개조식 복사본: 예전 청커는 1985자를 청크 1개로 만들었다.
    row = "| 지역 | 매장수 | 매출(억원) | 성장률 | 비고 |"
    table = " ".join([row] * 60)
    assert len(table) > 1900
    chunks = chunk_spans(table, max_chars=400)
    assert len(chunks) >= 5
    assert all(c.end - c.start <= 400 for c in chunks)
    assert chunks[0].start == 0 and chunks[-1].end == len(table)
    # 이웃 창은 겹친다(경계에 걸친 문장이 잘려 사라지지 않게)
    assert all(chunks[i + 1].start < chunks[i].end for i in range(len(chunks) - 1))


def test_decimal_points_do_not_split_sentences():
    s = "점유율은 3.5%로 올랐다. 다음 분기 전망은 2.1% 성장이다."
    texts = [c.text for c in chunk_spans(s, max_chars=20)]
    assert "점유율은 3.5%로 올랐다." in texts


def test_spans_point_into_original_text():
    s = "첫 문장이다.\n\n둘째 문장이다. 셋째 문장이다."
    for c in chunk_spans(s, max_chars=12):
        assert s[c.start : c.end].split() == c.text.split()


def test_excerpt_keeps_focus_with_neighbors_and_marks_omissions():
    sents = [f"{i}번째 문단의 일반적인 내용이다." for i in range(200)]
    sents[120] = "핵심: A사의 3분기 매출은 18% 늘었다."
    doc = " ".join(sents)
    chunks = chunk_spans(doc, max_chars=200)
    focus = next(i for i, c in enumerate(chunks) if "핵심" in c.text)
    ex = build_excerpt(doc, chunks, [focus], max_chars=1000)
    assert "핵심: A사의 3분기 매출은 18% 늘었다." in ex
    assert OMIT_MARK.strip() in ex
    assert len(ex) < 1200 < len(doc)
    # 발췌 부분은 원문의 연속 구간 그대로다(근거 대조가 그대로 통한다).
    for part in ex.split(OMIT_MARK.strip()):
        assert part.strip() in doc


def test_keyword_score_catches_acronym_that_embeddings_miss():
    nodes = {
        "1": "국내 매출 성장 / 지역별 매출 추이",
        "2": "고객 충성도 상승 / 재구매율",
        "3": "EBITDA 마진 개선 / EBITDA 마진 추이",
    }
    chunks = chunk_spans("회사의 EBITDA 마진은 2024년 18%로 개선됐다.")
    scores = keyword_scores(nodes, chunks)
    assert max(scores, key=lambda k: scores[k][0]) == "3"
    assert scores["3"][0] >= 0.25 and scores["2"][0] < 0.25


def test_node_embedding_text_includes_parent_path_and_questions():
    parent = {"node_id": "1", "parent_id": None, "hypothesis": "국내 사업이 성장한다"}
    child = {
        "node_id": "1.1", "parent_id": "1", "hypothesis": "출점이 매출을 늘린다",
        "sub_questions": ["신규 매장의 매출 기여는?"],
        "required_evidence": [{"description": "매장별 매출"}],
    }
    t = node_embed_text(child, [parent, child])
    assert t.startswith("상위: 국내 사업이 성장한다")
    assert "신규 매장의 매출 기여는?" in t and "매장별 매출" in t
