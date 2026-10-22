# analyze_content Eval 하네스

`analyze_content`의 판정 품질을 골드셋 기준으로 측정한다.

## 파일

- `documents.jsonl` — 평가에 쓰는 자료. `{doc_id, project_id, label, document}`.
- `goldset.jsonl` — **정답 라벨(당신이 검토·수정).** 한 줄 = (자료, 노드) 한 쌍.
  `{doc_id, node_id, gold_stance, gold_rationale_sentences, reviewed, note}`.
- `goldset.template.jsonl` — 포맷 예시.
- `run_eval.py` — 지표 계산기. `label_cli.py` — 대화형 라벨링 도구.

## 지표 (SciFact 방식)

1. **label-only F1** — stance(SUPPORTS/REFUTES/WEAK_RELATED/IRRELEVANT)만 맞으면 정답.
   4-class macro-F1 + accuracy + class별 P/R/F1.
2. **selection+label F1** — 더 엄격. stance가 맞고 **rationale 문장까지** 정확히
   집었을 때만 문장 단위 크레딧. stance가 틀리면 그 쌍의 문장은 전부 오답.

추가로 **3×3 혼동행렬**(행=정답, 열=예측)을 출력한다.

## 시드 상태 (중요)

`goldset.jsonl`의 초기 10쌍은 이전 5-자료 판정에서 나온 **모델 예측을 정답 후보로
넣어둔 것**이며 전부 `reviewed: false`다. 검토 전 점수는 (정답=예측이라) 순환적이므로
참고용일 뿐이다. **당신이 각 라벨을 검토·수정하고 `reviewed: true`로 바꾼 뒤**의
점수가 진짜 신호다. run_eval은 미검토 쌍 수를 경고로 알려준다.

## 라벨링 방법

### 방법 A — 대화형 CLI (권장)
```bash
make label
```
자료의 문장이 번호와 함께 표시되고, 노드 가설 목록이 나온다. `node_id` → `stance`
(s=지지, r=반증, w=약한 관련, i=무관) → (지지/반증이면) rationale 문장 번호를 입력하면 `goldset.jsonl`에 upsert된다.

### 방법 B — 파일 직접 편집
`goldset.jsonl`을 열어 각 줄의 `gold_stance` / `gold_rationale_sentences`를 고치고
`reviewed`를 `true`로 바꾼다. `gold_rationale_sentences`는 **원문 문장을 그대로 복사**
(WEAK_RELATED/IRRELEVANT면 `[]`). 새 쌍은 `goldset.template.jsonl` 포맷대로 줄을 추가한다.

30~50쌍을 목표로 다양한 자료·노드·stance를 담으면 지표가 안정된다.

## 실행
```bash
make eval
```
(내부적으로 db+api를 띄우고 `python eval/run_eval.py`를 돌린다. 임베딩은 캐시되며
Voyage 무료 등급 3 RPM 제한에 맞춰 백오프가 자동 페이싱한다.)
