"use client";

import { useState } from "react";
import { StanceBadge } from "./StanceBadge";

const FACTOR_LABEL: Record<string, string> = {
  evidence_type_fit: "증거유형 적합",
  source_credibility: "출처 신뢰",
  recency: "최신성",
  scale: "규모/표본",
};

type Matched = {
  node_id: string;
  hypothesis: string;
  candidate_similarity: number;
  stance: string;
  rationale_sentences: string[];
  warrant: string;
  provided_evidence_types: string[];
  strength: {
    score: number;
    factors: Record<string, number | null>;
    not_assessed: string[];
    band?: string;
    assessed_count?: number;
  };
  rationale_dropped?: number;
  rationale_repaired?: number;
  downgraded?: boolean;
  candidate_via?: string;
};

type Failed = { node_id: string; hypothesis: string; code: string; message: string };

const FAIL_REASON: Record<string, string> = {
  llm_overloaded: "AI 서비스 혼잡",
  llm_rate_limited: "AI 서비스 요청 한도",
  llm_circuit_open: "AI 서비스 장애로 호출 중단",
  budget_exceeded: "오늘 분석 한도 소진",
  llm_schema_invalid: "AI 응답 형식 오류",
  llm_output_truncated: "AI 응답이 잘림",
};

function bandOf(st: Matched["strength"]) {
  return st.band ?? (st.score >= 0.7 ? "강" : st.score >= 0.4 ? "중" : st.score > 0 ? "약" : "없음");
}

function StrengthView({ strength }: { strength: Matched["strength"] }) {
  const pct = Math.round(strength.score * 100);
  const band = bandOf(strength);
  const assessed = strength.assessed_count ?? 4 - (strength.not_assessed?.length ?? 0);
  return (
    <div>
      <div className="sbar">
        <span className={`band b-${band}`}>근거 강도 {band}</span>
        <div className="track">
          <div className="fill" style={{ width: `${pct}%` }} />
        </div>
      </div>
      <div className="note" style={{ marginTop: 0 }}>
        아래 {assessed}개 요인으로 매긴 상대 등급입니다. 맞을 확률이 아닙니다(점수 {strength.score}).
        {assessed < 2 && " 평가할 수 있는 요인이 적어 등급을 '중'까지만 줍니다."}
      </div>
      <div className="factors">
        {Object.keys(FACTOR_LABEL).map((k) => {
          const v = strength.factors?.[k];
          const na = strength.not_assessed?.includes(k) || v === null || v === undefined;
          return (
            <span key={k} className={`factor ${na ? "na" : ""}`}>
              {FACTOR_LABEL[k]}: {na ? "미산정" : v}
            </span>
          );
        })}
      </div>
      {strength.not_assessed?.length > 0 && (
        <div className="note">
          ※ 미산정 요인({strength.not_assessed.map((k) => FACTOR_LABEL[k] ?? k).join(", ")})은
          자료에 정보가 없어 점수 계산에서 제외하고 나머지로 재정규화했습니다(0으로 채우지 않음).
        </div>
      )}
    </div>
  );
}

function Row({ m }: { m: Matched }) {
  const [open, setOpen] = useState(false);
  const isEvidence = m.stance === "SUPPORTS" || m.stance === "REFUTES";
  return (
    <details className={`result-row ${m.stance}`} open={open}>
      <summary
        onClick={(e) => {
          e.preventDefault();
          setOpen((o) => !o);
        }}
      >
        <StanceBadge stance={m.stance} />
        <span className="rhyp">{m.hypothesis}</span>
        {isEvidence && <span className="rstr">강도 {bandOf(m.strength)}</span>}
      </summary>
      <div className="rdetail">
        {isEvidence ? (
          <>
            <div className="vtag">원문 발췌 (요약·의역 아님)</div>
            {m.rationale_sentences.map((s, i) => (
              <blockquote key={i} className="rationale">
                {s}
              </blockquote>
            ))}
            {!!m.rationale_dropped && (
              <div className="note">
                AI가 제시한 근거 중 {m.rationale_dropped}개는 원문에서 찾을 수 없어 뺐습니다.
              </div>
            )}
            <div className="warrant">
              <b>왜 이 목표에 정렬되나 (AI 해석):</b> {m.warrant}
            </div>
            <StrengthView strength={m.strength} />
          </>
        ) : (
          <>
            <p className="muted" style={{ margin: 0, fontSize: 14 }}>
              {m.downgraded
                ? "AI는 지지/반증으로 봤지만, 그 근거 문장을 원문에서 확인하지 못해 '약한 관련'으로 낮췄습니다. 원문을 직접 확인해 보세요."
                : m.stance === "WEAK_RELATED"
                ? "주제·맥락은 관련되지만 지지/반증을 확정할 증거가 부족합니다. 근거 문장은 제시하지 않습니다."
                : "이 가설과 실질적 관련이 없습니다."}
            </p>
            <div className="warrant">
              <b>판정 사유:</b> {m.warrant}
            </div>
          </>
        )}
      </div>
    </details>
  );
}

const FILTERS = [
  { key: "all", label: "전체" },
  { key: "aligned", label: "정렬됨" },
  { key: "WEAK_RELATED", label: "약한 관련" },
  { key: "IRRELEVANT", label: "무관" },
];

export function AnalysisResult({ result }: { result: any }) {
  const nodes: Matched[] = result?.matched_nodes ?? [];
  const failed: Failed[] = result?.failed_nodes ?? [];
  const [filter, setFilter] = useState("all");

  const count = (fn: (m: Matched) => boolean) => nodes.filter(fn).length;
  const nSup = count((m) => m.stance === "SUPPORTS");
  const nRef = count((m) => m.stance === "REFUTES");
  const nWeak = count((m) => m.stance === "WEAK_RELATED");
  const nIrr = count((m) => m.stance === "IRRELEVANT");
  const countFor: Record<string, number> = {
    all: nodes.length,
    aligned: nSup + nRef,
    WEAK_RELATED: nWeak,
    IRRELEVANT: nIrr,
  };

  const order: Record<string, number> = { SUPPORTS: 0, REFUTES: 0, WEAK_RELATED: 1, IRRELEVANT: 2 };
  const visible = nodes
    .filter((m) => {
      if (filter === "all") return true;
      if (filter === "aligned") return m.stance === "SUPPORTS" || m.stance === "REFUTES";
      return m.stance === filter;
    })
    .sort((a, b) => order[a.stance] - order[b.stance] || b.strength.score - a.strength.score);

  const parts: string[] = [];
  if (nSup) parts.push(`${nSup}개 목표에 지지 근거`);
  if (nRef) parts.push(`${nRef}개 목표에 반증 근거`);

  return (
    <div>
      <div className="align-summary">
        {nSup + nRef > 0 ? (
          <>
            이 자료는 <b>{parts.join(", ")}</b>를 줍니다.
            {nWeak + nIrr > 0 && <span className="sub"> · 그 외 약한 관련 {nWeak}, 무관 {nIrr}</span>}
          </>
        ) : (
          <>
            이 자료는 아직 <b>어느 목표에도 지지/반증 근거를 주지 못했습니다.</b>
            <span className="sub"> · 약한 관련 {nWeak}, 무관 {nIrr}</span>
          </>
        )}
      </div>

      <div className="chips" style={{ marginBottom: "var(--sp-4)" }}>
        {FILTERS.map((f) => (
          <button
            key={f.key}
            className={`chip ${filter === f.key ? "selected" : ""} ${countFor[f.key] === 0 ? "disabled" : ""}`}
            onClick={() => countFor[f.key] > 0 && setFilter(f.key)}
          >
            {f.label}
            <span className="count">{countFor[f.key]}</span>
          </button>
        ))}
      </div>

      {visible.map((m) => (
        <Row key={m.node_id} m={m} />
      ))}

      {failed.length > 0 && (
        <div className="failbox">
          <b>판정하지 못한 목표 {failed.length}개</b> (무관하다는 뜻이 아닙니다). 같은 자료를 다시 넣으면 이
          목표들만 다시 판정합니다.
          <ul>
            {failed.map((f) => (
              <li key={f.node_id}>
                {f.hypothesis} · {FAIL_REASON[f.code] ?? "처리 오류"}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="provenance">
        {result?.input_mode === "excerpt" &&
          "자료가 길어 목표마다 관련된 부분과 그 앞뒤만 발췌해 판정했습니다. 근거 문장은 전체 원문과 대조했습니다. "}
        판정과 해석은 AI가 만든 것이고, 근거 문장만 원문과 대조해 확인했습니다. 출처 자체의 신뢰성은
        검증하지 않습니다.
        {result?.model && ` · 모델 ${result.model}`}
        {result?.prompt_version && ` · 판정 기준 ${result.prompt_version}`}
      </div>
    </div>
  );
}
